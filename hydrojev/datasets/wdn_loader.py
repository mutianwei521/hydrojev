"""Load the real EPANET and benchmark assets bundled with the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
import pandas as pd

from hydrojev.config import HydroJEVConfig, load_config

if TYPE_CHECKING:
    from wntr.network import WaterNetworkModel


class DatasetUnavailableError(FileNotFoundError):
    """A named benchmark cannot be run from the locally available assets."""

    def __init__(self, message: str, *, dataset: str, status: str = "not_run") -> None:
        super().__init__(message)
        self.dataset = dataset
        self.status = status


@dataclass(frozen=True)
class NetworkBundle:
    name: str
    model: "WaterNetworkModel"
    source_path: Path
    junctions: tuple[str, ...]
    reservoirs: tuple[str, ...]
    tanks: tuple[str, ...]
    pipes: tuple[str, ...]
    pumps: tuple[str, ...]
    valves: tuple[str, ...]
    protected_nodes: tuple[str, ...]
    hydraulic_timestep_s: int
    headloss_model: str


@dataclass(frozen=True)
class TimeSeriesBundle:
    name: str
    frame: pd.DataFrame
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    timestamp_column: str
    label_column: str | None
    feature_groups: Mapping[str, tuple[str, ...]]
    sample_interval_s: float
    source_path: Path

    @property
    def feature_columns(self) -> tuple[str, ...]:
        excluded = {self.timestamp_column}
        if self.label_column:
            excluded.add(self.label_column)
        return tuple(column for column in self.frame.columns if column not in excluded)


def _resolve_config(config: HydroJEVConfig | None) -> HydroJEVConfig:
    return config if config is not None else load_config()


def _wntr_net3_path() -> Path:
    import wntr

    path = Path(wntr.__file__).resolve().parent / "library" / "networks" / "Net3.inp"
    if not path.is_file():
        raise FileNotFoundError(f"WNTR library Net3.inp not found at {path}")
    return path


def load_network(name: str, *, config: HydroJEVConfig | None = None) -> NetworkBundle:
    """Load Net1, Net3, or CTOWN and expose topology/provenance metadata."""

    import wntr

    cfg = _resolve_config(config)
    canonical = name.strip().lower()
    candidates = {
        "net1": cfg.paths.ref_code / "FDI-WDNs" / "Net1.inp",
        "net3": _wntr_net3_path(),
        "ctown": cfg.paths.ref_code / "BATADAL" / "CTOWN.INP",
    }
    if canonical not in candidates:
        raise ValueError("network name must be Net1, Net3, or CTOWN")
    source = candidates[canonical].resolve()
    if not source.is_file():
        raise FileNotFoundError(f"EPANET network file not found: {source}")

    model = wntr.network.WaterNetworkModel(str(source))
    pump_endpoints: set[str] = set()
    for pump_name in model.pump_name_list:
        pump = model.get_link(pump_name)
        pump_endpoints.update((pump.start_node_name, pump.end_node_name))
    protected = set(model.reservoir_name_list) | set(model.tank_name_list) | pump_endpoints

    return NetworkBundle(
        name={"net1": "Net1", "net3": "Net3", "ctown": "CTOWN"}[canonical],
        model=model,
        source_path=source,
        junctions=tuple(model.junction_name_list),
        reservoirs=tuple(model.reservoir_name_list),
        tanks=tuple(model.tank_name_list),
        pipes=tuple(model.pipe_name_list),
        pumps=tuple(model.pump_name_list),
        valves=tuple(model.valve_name_list),
        protected_nodes=tuple(sorted(protected)),
        hydraulic_timestep_s=int(model.options.time.hydraulic_timestep),
        headloss_model=str(model.options.hydraulic.headloss),
    )


def classify_batadal_columns(columns: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Classify BATADAL signals by their documented naming convention."""

    values = tuple(columns)
    patterns = {
        "tank_level": "L_T",
        "pump_flow": "F_PU",
        "pump_status": "S_PU",
        "valve_flow": "F_V",
        "valve_status": "S_V",
        "junction_pressure": "P_J",
    }
    return {
        group: tuple(column for column in values if column.upper().startswith(prefix))
        for group, prefix in patterns.items()
    }


def chronological_split(
    frame: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split without shuffling, retaining original indices for leakage audits."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if len(frame) < 2:
        raise ValueError("at least two observations are required")
    boundary = int(len(frame) * train_fraction)
    boundary = min(max(boundary, 1), len(frame) - 1)
    return frame.iloc[:boundary].copy(), frame.iloc[boundary:].copy()


def infer_sample_interval_s(timestamps: pd.Series) -> float:
    """Infer the modal positive observation interval from parsed timestamps."""

    parsed = pd.to_datetime(timestamps, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"timestamp parsing failed for {bad} observations")
    differences = parsed.diff().dt.total_seconds().dropna()
    differences = differences[differences > 0]
    if differences.empty:
        raise ValueError("no positive timestamp interval is available")
    modes = differences.mode()
    return float(modes.iloc[0] if not modes.empty else differences.median())


def _normalize_attack_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    normalized = numeric.notna() & numeric.ne(0)
    text = values.astype(str).str.strip().str.lower()
    positive_words = {"attack", "attacked", "anomaly", "true", "yes"}
    normalized |= text.isin(positive_words)
    return normalized.astype(np.int8)


def _find_batadal_path(ref_code: Path, variant: str) -> Path:
    aliases = {
        "dataset03": "dataset03.csv",
        "normal": "dataset03.csv",
        "dataset04": "dataset04.csv",
        "training_attack": "dataset04.csv",
        "test_dataset": "test_dataset.csv",
        "test": "test_dataset.csv",
    }
    key = variant.strip().lower().removesuffix(".csv")
    if key.startswith("batadal_"):
        key = key.removeprefix("batadal_")
    if key not in aliases:
        raise ValueError(
            "BATADAL variant must be dataset03, dataset04, or test_dataset"
        )
    filename = aliases[key]
    candidates = (
        ref_code / "aeed" / "data" / filename,
        ref_code / "BATADAL" / f"BATADAL_{filename}",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"BATADAL file {filename} was not found under {ref_code}")


def load_batadal(
    variant: str = "dataset03",
    *,
    config: HydroJEVConfig | None = None,
    train_fraction: float | None = None,
) -> TimeSeriesBundle:
    """Load a local BATADAL CSV and infer its actual observation cadence."""

    cfg = _resolve_config(config)
    source = _find_batadal_path(cfg.paths.ref_code, variant)
    frame = pd.read_csv(source, low_memory=False)
    timestamp_column = "DATETIME"
    if timestamp_column not in frame:
        raise ValueError(f"BATADAL file lacks {timestamp_column}: {source}")
    frame[timestamp_column] = pd.to_datetime(
        frame[timestamp_column].astype(str).str.strip(),
        format="%d/%m/%y %H",
        errors="raise",
    )
    frame = frame.sort_values(timestamp_column, kind="stable")
    if "ATT_FLAG" in frame:
        frame["ATT_FLAG"] = _normalize_attack_labels(frame["ATT_FLAG"])
    interval = infer_sample_interval_s(frame[timestamp_column])
    fraction = cfg.experiments.train_fraction if train_fraction is None else train_fraction
    train, validation = chronological_split(frame, fraction)
    return TimeSeriesBundle(
        name=f"BATADAL:{source.stem}",
        frame=frame,
        train_frame=train,
        validation_frame=validation,
        timestamp_column=timestamp_column,
        label_column="ATT_FLAG" if "ATT_FLAG" in frame else None,
        feature_groups=classify_batadal_columns(frame.columns),
        sample_interval_s=interval,
        source_path=source,
    )


def _classify_wadi_columns(columns: Iterable[str]) -> dict[str, tuple[str, ...]]:
    candidates = tuple(columns)
    actuator_tokens = ("PUMP", "MV", "VALVE", "STATUS", "_P_")
    actuators = tuple(
        column
        for column in candidates
        if any(token in column.upper() for token in actuator_tokens)
    )
    actuator_set = set(actuators)
    sensors = tuple(column for column in candidates if column not in actuator_set)
    return {"sensor": sensors, "actuator": actuators}


def load_wadi(
    *,
    config: HydroJEVConfig | None = None,
    train_fraction: float | None = None,
) -> TimeSeriesBundle:
    """Load WADI if the gated benchmark has been supplied by the user."""

    cfg = _resolve_config(config)
    root = cfg.paths.ref_code / "WADI"
    files = sorted(root.rglob("*.csv")) if root.is_dir() else []
    if not files:
        raise DatasetUnavailableError(
            f"WADI benchmark files are absent from {root}; obtain authorized data "
            "before running the named WADI experiment",
            dataset="WADI",
        )

    source = next(
        (path for path in files if "attack" in path.name.lower()), files[0]
    ).resolve()
    frame = pd.read_csv(source, low_memory=False)
    lower_names = {column.lower().strip(): column for column in frame.columns}
    label_column = next(
        (
            original
            for lowered, original in lower_names.items()
            if "attack" in lowered or "label" in lowered
        ),
        None,
    )
    if label_column:
        frame[label_column] = _normalize_attack_labels(frame[label_column])

    date_column = next((value for key, value in lower_names.items() if key == "date"), None)
    time_column = next((value for key, value in lower_names.items() if key == "time"), None)
    datetime_column = next(
        (value for key, value in lower_names.items() if "datetime" in key), None
    )
    timestamp_column = "__timestamp__"
    if datetime_column:
        timestamps = pd.to_datetime(frame[datetime_column], errors="coerce")
    elif date_column and time_column:
        timestamps = pd.to_datetime(
            frame[date_column].astype(str) + " " + frame[time_column].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    else:
        raise ValueError(f"could not identify WADI timestamp columns in {source}")
    frame.insert(0, timestamp_column, timestamps)
    frame = frame.sort_values(timestamp_column, kind="stable")
    interval = infer_sample_interval_s(frame[timestamp_column])
    fraction = cfg.experiments.train_fraction if train_fraction is None else train_fraction
    train, validation = chronological_split(frame, fraction)
    excluded = {
        timestamp_column,
        label_column,
        date_column,
        time_column,
        datetime_column,
    }
    feature_columns = [column for column in frame.columns if column not in excluded]
    return TimeSeriesBundle(
        name="WADI",
        frame=frame,
        train_frame=train,
        validation_frame=validation,
        timestamp_column=timestamp_column,
        label_column=label_column,
        feature_groups=_classify_wadi_columns(feature_columns),
        sample_interval_s=interval,
        source_path=source,
    )
