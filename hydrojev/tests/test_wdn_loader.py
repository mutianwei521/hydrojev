from pathlib import Path

import pandas as pd
import pytest

from hydrojev.config import load_config
from hydrojev.datasets.wdn_loader import (
    DatasetUnavailableError,
    chronological_split,
    classify_batadal_columns,
    load_batadal,
    load_network,
    load_wadi,
)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "junction_count", "pump_count", "hydraulic_step"),
    [
        ("Net1", 9, 1, 3600),
        ("Net3", 92, 2, 3600),
        ("CTOWN", 388, 11, 900),
    ],
)
def test_load_network_resolves_real_epanet_assets(
    project_root: Path,
    name: str,
    junction_count: int,
    pump_count: int,
    hydraulic_step: int,
) -> None:
    bundle = load_network(name, config=load_config(project_root=project_root))

    assert len(bundle.junctions) == junction_count
    assert len(bundle.pumps) == pump_count
    assert bundle.hydraulic_timestep_s == hydraulic_step
    assert bundle.headloss_model.upper() == "H-W"
    assert bundle.source_path.is_file()
    assert set(bundle.reservoirs + bundle.tanks).issubset(bundle.protected_nodes)
    assert bundle.protected_nodes


def test_unknown_network_name_is_rejected(project_root: Path) -> None:
    with pytest.raises(ValueError, match="Net1, Net3, or CTOWN"):
        load_network("imaginary", config=load_config(project_root=project_root))


def test_batadal_schema_and_sampling_are_inferred_from_local_csv(project_root: Path) -> None:
    bundle = load_batadal(
        "dataset03", config=load_config(project_root=project_root), train_fraction=0.8
    )

    assert len(bundle.frame) == 8761
    assert bundle.sample_interval_s == 3600.0
    assert bundle.label_column == "ATT_FLAG"
    assert len(bundle.feature_groups["tank_level"]) == 7
    assert len(bundle.feature_groups["pump_flow"]) == 11
    assert len(bundle.feature_groups["pump_status"]) == 11
    assert len(bundle.feature_groups["junction_pressure"]) == 12
    assert bundle.train_frame.index.max() < bundle.validation_frame.index.min()
    assert len(bundle.train_frame) == int(0.8 * len(bundle.frame))
    assert set(bundle.frame["ATT_FLAG"].unique()) == {0}


def test_schema_classifier_does_not_confuse_flow_and_status_columns() -> None:
    groups = classify_batadal_columns(
        ["L_T1", "F_PU1", "S_PU1", "F_V2", "S_V2", "P_J280", "ATT_FLAG"]
    )

    assert groups == {
        "tank_level": ("L_T1",),
        "pump_flow": ("F_PU1",),
        "pump_status": ("S_PU1",),
        "valve_flow": ("F_V2",),
        "valve_status": ("S_V2",),
        "junction_pressure": ("P_J280",),
    }


def test_chronological_split_never_shuffles_or_overlaps() -> None:
    frame = pd.DataFrame({"value": range(10)})

    train, validation = chronological_split(frame, 0.8)

    assert train["value"].tolist() == list(range(8))
    assert validation["value"].tolist() == [8, 9]
    assert train.index.intersection(validation.index).empty


def test_missing_wadi_is_reported_as_not_run(project_root: Path) -> None:
    with pytest.raises(DatasetUnavailableError) as caught:
        load_wadi(config=load_config(project_root=project_root))

    assert caught.value.status == "not_run"
    assert "WADI" in str(caught.value)
    assert "synthetic" not in str(caught.value).lower()

