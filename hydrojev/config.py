"""Typed configuration loading with secret-safe serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration violates a scientific or safety invariant."""


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    ref_code: Path
    ref_paper: Path
    artifacts: Path


@dataclass(frozen=True)
class ClusteringConfig:
    alpha: float = 0.35
    beta: float = 0.45
    threshold: float = 0.65
    acoustic_speed_m_s: float = 1000.0
    default_flow_m3_s: float = 0.01
    darcy_friction_factor: float = 0.02


@dataclass(frozen=True)
class SurrogateConfig:
    sequence_length: int = 12
    hidden_size: int = 20
    num_layers: int = 2
    dense_size: int = 60
    learning_rate: float = 1e-3
    minimum_learning_rate: float = 1e-6
    max_epochs: int = 100
    batch_size: int = 128
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class ConcealmentConfig:
    max_modified_features: int = 4
    threshold_quantile: float = 0.995
    budget: int = 200
    patience: int = 15
    candidate_values: int = 21
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class FSFDIConfig:
    max_modified_measurements: int = 4
    piecewise_segments: int = 8
    maximum_flow_fraction: float = 0.20
    maximum_head_fraction: float = 0.10
    maximum_demand_fraction: float = 0.20
    state_deviation_limit: float = 0.25
    cusum_threshold: float = 5.0
    chi_square_threshold: float = 9.21
    solver_order: tuple[str, ...] = ("gurobi", "highs", "pulp")
    solver_time_limit_s: float = 30.0


@dataclass(frozen=True)
class ResidualConfig:
    covariance_regularization: float = 1e-6
    minimum_rat_window: int = 8
    cpdz_alert_threshold: float = 3.0
    gefm_alert_threshold: float = 0.15
    rat_alert_threshold: float = 0.25


@dataclass(frozen=True)
class JevConfig:
    endpoint: str = "https://api.typesafe.ai/v1/systemone"
    model: str = "jev-latest"
    timeout_s: float = 10.0
    max_attempts: int = 3
    backoff_initial_s: float = 0.25
    mock_base_url: str = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class SafetyConfig:
    maximum_state_age_s: float = 2.0
    minimum_evidence_count: int = 3
    minimum_noul_probability: float = 0.80
    minimum_attack_persistence: int = 2
    require_local_hydraulic_violation: bool = True
    require_operator_confirmation: bool = False
    isolation_hysteresis_steps: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    train_fraction: float = 0.80
    latency_iterations: int = 100
    recall_target: float = 0.95
    maximum_energy_increase_fraction: float = 0.05
    maximum_total_latency_ms: float = 150.0


@dataclass(frozen=True)
class HydroJEVConfig:
    paths: PathsConfig
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    concealment: ConcealmentConfig = field(default_factory=ConcealmentConfig)
    fs_fdi: FSFDIConfig = field(default_factory=FSFDIConfig)
    residuals: ResidualConfig = field(default_factory=ResidualConfig)
    jev: JevConfig = field(default_factory=JevConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_public_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation that can never include credentials."""

        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _reject_secrets(mapping: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        normalized = str(key).lower().replace("-", "_")
        if normalized in {"api_key", "apikey", "access_token", "secret", "password"}:
            raise ConfigError(
                f"API key or other secret is not allowed in configuration ({dotted}); "
                "use a process environment variable"
            )
        if isinstance(value, Mapping):
            _reject_secrets(value, dotted)


def _validate(config: HydroJEVConfig) -> None:
    clustering = config.clustering
    if not (0.0 <= clustering.alpha <= 1.0 and 0.0 <= clustering.beta <= 1.0):
        raise ConfigError("clustering alpha and beta must each be in [0, 1]")
    if clustering.alpha + clustering.beta > 1.0 + 1e-12:
        raise ConfigError("clustering alpha + beta must be <= 1")
    if clustering.acoustic_speed_m_s <= 0:
        raise ConfigError("clustering acoustic_speed_m_s must be positive")
    if not (1 <= config.concealment.max_modified_features <= 4):
        raise ConfigError("concealment max_modified_features must be between 1 and 4")
    if not (0.0 < config.concealment.threshold_quantile < 1.0):
        raise ConfigError("concealment threshold_quantile must be in (0, 1)")
    if not (0.0 < config.experiments.train_fraction < 1.0):
        raise ConfigError("experiments train_fraction must be in (0, 1)")
    if config.jev.max_attempts < 1 or config.jev.timeout_s <= 0:
        raise ConfigError("Jev retry count and timeout must be positive")


def load_config(
    config_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> HydroJEVConfig:
    """Load defaults plus an optional YAML override and validate invariants."""

    resolved_root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    default_path = Path(__file__).resolve().parent / "configs" / "default_config.yaml"
    with default_path.open("r", encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}

    override: dict[str, Any] = {}
    if config_path is not None:
        with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ConfigError("configuration root must be a mapping")
        override = dict(loaded)
    _reject_secrets(override)
    raw = _deep_merge(defaults, override)

    path_values = raw.pop("paths", {})
    paths = PathsConfig(
        project_root=resolved_root,
        ref_code=(resolved_root / path_values.get("ref_code", "refCode")).resolve(),
        ref_paper=(resolved_root / path_values.get("ref_paper", "refPaper")).resolve(),
        artifacts=(resolved_root / path_values.get("artifacts", "artifacts")).resolve(),
    )
    try:
        config = HydroJEVConfig(
            paths=paths,
            clustering=ClusteringConfig(**raw.get("clustering", {})),
            surrogate=SurrogateConfig(**raw.get("surrogate", {})),
            concealment=ConcealmentConfig(**raw.get("concealment", {})),
            fs_fdi=FSFDIConfig(
                **{
                    **raw.get("fs_fdi", {}),
                    "solver_order": tuple(
                        raw.get("fs_fdi", {}).get(
                            "solver_order", FSFDIConfig().solver_order
                        )
                    ),
                }
            ),
            residuals=ResidualConfig(**raw.get("residuals", {})),
            jev=JevConfig(**raw.get("jev", {})),
            safety=SafetyConfig(**raw.get("safety", {})),
            experiments=ExperimentConfig(**raw.get("experiments", {})),
        )
    except TypeError as exc:
        raise ConfigError(f"unknown or malformed configuration field: {exc}") from exc
    _validate(config)
    return config

