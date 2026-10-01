"""Tests for the async Jev client: retries, auth, validation, and fallback."""

from __future__ import annotations

import asyncio
import random

import httpx
import pytest

from hydrojev.arbiter.decision_primitives import JevSource, ThreatCause
from hydrojev.arbiter.jev_client import (
    JevAuthError,
    JevClient,
    JevKeyMissingError,
    JevUnavailableError,
    JevValidationError,
    RetryPolicy,
)
from hydrojev.simulation.mock_jev_server import arbitrate_state

KEY = "unit-test-dummy-key"  # not a real credential
SAMPLE_STATE = {
    "detectors": {"cpdz_residual_normalized": {"state": "alerting"}, "vectorized_cusum": {"state": "nominal"}},
    "hydraulic_consistency": {"within_tolerance": True, "extra_energy_fraction": 0.6},
    "freshness": {"is_fresh": True},
}
OK_BODY = arbitrate_state(SAMPLE_STATE)


def _run(coro):
    return asyncio.run(coro)


async def _noop_sleep(_delay: float) -> None:
    return None


def _client(handler, *, sleep=_noop_sleep, retry_policy=None, **kwargs) -> JevClient:
    return JevClient(
        endpoint="https://mock.local/v1/systemone",
        transport=httpx.MockTransport(handler),
        sleep=sleep,
        retry_policy=retry_policy or RetryPolicy(max_attempts=4, base_delay_s=0.5),
        rng=random.Random(0),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Happy path and auth handling
# --------------------------------------------------------------------------- #


def test_decide_returns_live_decision_and_sends_bearer_key(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        return httpx.Response(200, json=OK_BODY)

    decision = _run(_client(handler).decide(SAMPLE_STATE))
    assert decision.source is JevSource.LIVE
    assert decision.attempts == 1
    assert "network_round_trip_ms" in decision.timing_ms
    assert decision.threat_cause.as_threat_cause() is ThreatCause.HYDRAULIC_FS_FDI
    # The Bearer key is transmitted, exactly once, in the Authorization header.
    assert captured["auth"] == f"Bearer {KEY}"


def test_missing_key_raises_before_any_request(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=OK_BODY)

    with pytest.raises(JevKeyMissingError):
        _run(_client(handler).decide(SAMPLE_STATE))
    assert calls["n"] == 0  # never hit the network without a key


def test_403_is_auth_error_and_not_retried(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    with pytest.raises(JevAuthError):
        _run(_client(handler).decide(SAMPLE_STATE))
    assert calls["n"] == 1


def test_422_is_validation_error_and_not_retried(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(422, json={"error": {"message": "bad field"}})

    with pytest.raises(JevValidationError):
        _run(_client(handler).decide(SAMPLE_STATE))
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Retry behavior
# --------------------------------------------------------------------------- #


def test_429_then_200_retries_and_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    statuses = [429, 200]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = calls["n"]
        calls["n"] += 1
        if statuses[i] == 200:
            return httpx.Response(200, json=OK_BODY)
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    decision = _run(_client(handler).decide(SAMPLE_STATE))
    assert decision.attempts == 2
    assert calls["n"] == 2


def test_persistent_529_exhausts_retries(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(529, json={"error": {"message": "overloaded"}})

    with pytest.raises(JevUnavailableError):
        _run(_client(handler, retry_policy=RetryPolicy(max_attempts=3)).decide(SAMPLE_STATE))
    assert calls["n"] == 3


def test_backoff_delays_increase(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"error": {"message": "overloaded"}})

    with pytest.raises(JevUnavailableError):
        _run(_client(handler, sleep=record_sleep, retry_policy=RetryPolicy(max_attempts=4)).decide(SAMPLE_STATE))
    # One sleep between each of the four attempts (three gaps), strictly increasing.
    assert len(delays) == 3
    assert delays[0] < delays[1] < delays[2]


def test_network_error_is_redacted_and_wrapped(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect to 10.0.0.1 with secret in url failed")

    with pytest.raises(JevUnavailableError) as excinfo:
        _run(_client(handler, retry_policy=RetryPolicy(max_attempts=2)).decide(SAMPLE_STATE))
    message = str(excinfo.value)
    assert "ConnectError" in message  # only the type is surfaced
    assert "secret" not in message and KEY not in message


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #


def test_fallback_used_when_no_key_configured(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - unused
        return httpx.Response(200, json=OK_BODY)

    decision = _run(_client(handler).decide_with_fallback(SAMPLE_STATE))
    assert decision.source is JevSource.LOCAL_FALLBACK
    assert decision.threat_cause.as_threat_cause() is ThreatCause.HYDRAULIC_FS_FDI


def test_fallback_used_when_service_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, json={"error": {"message": "overloaded"}})

    decision = _run(_client(handler, retry_policy=RetryPolicy(max_attempts=2)).decide_with_fallback(SAMPLE_STATE))
    assert decision.source is JevSource.LOCAL_FALLBACK


def test_rejected_key_is_not_masked_by_fallback(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "forbidden"}})

    # A present-but-rejected key is a real misconfiguration, surfaced not hidden.
    with pytest.raises(JevAuthError):
        _run(_client(handler).decide_with_fallback(SAMPLE_STATE))
