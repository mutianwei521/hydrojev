"""Async TypeSafe Jev client with bounded retries and an offline fallback.

Security invariants (Task 8, and the standing project constraint):

* The API key is read from the process environment **only at call time** and is
  never stored on the client, logged, or echoed into an exception message.
* Authentication (401/403) and validation (422) failures are never retried;
  the live gateway returns 403 for a missing/invalid key even though the docs
  list 401, so both are treated as auth failures.
* Rate-limit (429), overload (529), other 5xx, and network errors are retried
  with exponential backoff plus jitter, up to a bounded number of attempts.

When no key is configured or the remote call exhausts retries,
:meth:`JevClient.decide_with_fallback` degrades to the deterministic local mock
so the closed loop still runs fully offline. A present-but-rejected key raises,
so a real misconfiguration is surfaced rather than silently masked.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

import httpx

from hydrojev.arbiter.decision_primitives import (
    DEFAULT_MODEL,
    JevDecision,
    JevSchemaError,
    JevSource,
    build_request,
    parse_response,
)
from hydrojev.simulation.mock_jev_server import decide_locally

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"

NON_RETRYABLE_STATUS = frozenset({401, 403, 422})
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})


class JevError(RuntimeError):
    """Base class for Jev client failures."""


class JevAuthError(JevError):
    """Authentication was rejected (HTTP 401/403)."""


class JevKeyMissingError(JevAuthError):
    """No API key was present in the environment."""


class JevValidationError(JevError):
    """The request failed server-side validation (HTTP 422)."""


class JevUnavailableError(JevError):
    """The service was unreachable or overloaded after all retries."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    multiplier: float = 2.0
    jitter_fraction: float = 0.5


class JevClient:
    """Async client for the TypeSafe ``/v1/systemone`` three-question contract."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        api_key_env: str = "TYPESAFE_API_KEY",
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.retry = retry_policy or RetryPolicy()
        self._transport = transport
        self._api_key_env = api_key_env
        self._sleep = sleep
        self._rng = rng or random.Random()

    def _read_key(self) -> str:
        key = os.environ.get(self._api_key_env, "").strip()
        if not key:
            raise JevKeyMissingError(f"{self._api_key_env} is not set in the process environment")
        return key

    def _backoff_delay(self, attempt: int) -> float:
        base = min(self.retry.base_delay_s * (self.retry.multiplier ** (attempt - 1)), self.retry.max_delay_s)
        return base + base * self.retry.jitter_fraction * self._rng.random()

    @staticmethod
    def _raise_for_non_retryable(status: int) -> None:
        if status in (401, 403):
            raise JevAuthError(f"authentication failed (HTTP {status}); check the API key")
        if status == 422:
            raise JevValidationError("request failed server-side validation (HTTP 422)")

    async def decide(self, state: Mapping[str, Any]) -> JevDecision:
        """Call the live Jev service and return a validated decision.

        Raises :class:`JevKeyMissingError` if no key is configured,
        :class:`JevAuthError`/:class:`JevValidationError` on non-retryable
        rejections, and :class:`JevUnavailableError` if retries are exhausted.
        """

        key = self._read_key()
        request = build_request(state, model=self.model)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        last_error: JevUnavailableError | None = None
        async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout_s) as client:
            for attempt in range(1, self.retry.max_attempts + 1):
                start = time.perf_counter_ns()
                try:
                    response = await client.post(self.endpoint, json=request, headers=headers)
                except httpx.HTTPError as exc:
                    # Redacted: only the exception type, never the message (which
                    # could contain the URL/headers), reaches the error.
                    last_error = JevUnavailableError(f"transport error: {type(exc).__name__}")
                    if attempt < self.retry.max_attempts:
                        await self._sleep(self._backoff_delay(attempt))
                        continue
                    raise last_error
                round_trip_ms = (time.perf_counter_ns() - start) / 1_000_000.0

                if response.status_code == 200:
                    try:
                        body = response.json()
                    except ValueError as exc:
                        raise JevSchemaError("response body was not valid JSON") from exc
                    return parse_response(
                        body,
                        source=JevSource.LIVE,
                        timing_ms={"network_round_trip_ms": round_trip_ms},
                        attempts=attempt,
                    )

                if response.status_code in NON_RETRYABLE_STATUS:
                    self._raise_for_non_retryable(response.status_code)

                if response.status_code in RETRYABLE_STATUS:
                    last_error = JevUnavailableError(f"service returned HTTP {response.status_code}")
                    if attempt < self.retry.max_attempts:
                        await self._sleep(self._backoff_delay(attempt))
                        continue
                    raise last_error

                # Any other status (e.g. 400/404/409) is a non-retryable client error.
                raise JevError(f"unexpected response status HTTP {response.status_code}")

        raise last_error or JevUnavailableError("exhausted retries without a response")

    async def decide_with_fallback(self, state: Mapping[str, Any]) -> JevDecision:
        """Try the live service; fall back to the local mock when offline.

        Falls back only when no key is configured or the service is unavailable
        after retries. A rejected key (401/403) or a malformed request (422) is
        re-raised so a real misconfiguration is not silently masked.
        """

        try:
            return await self.decide(state)
        except (JevKeyMissingError, JevUnavailableError):
            return decide_locally(state, model=self.model, source=JevSource.LOCAL_FALLBACK)
