"""Contract tests for causal HydroJEV evidence encoding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hydrojev.benchmarks.detection_baselines import (
    fit_mahalanobis_detector,
    fit_pca_residual_detector,
)
from hydrojev.methods.evidence_encoder import HydroJEVEvidenceEncoder
from hydrojev.physics.surrogate_models import Standardizer


def _bundle(clean: np.ndarray) -> SimpleNamespace:
    standardizer = Standardizer.fit(clean)

    def predict(windows: np.ndarray) -> np.ndarray:
        return windows[:, -1, :]

    clean_std = standardizer.transform(clean)
    residuals = clean_std[12:] - clean_std[11:-1]
    scale = np.std(residuals, axis=0) + 0.1
    cpdz = np.sqrt(np.mean(np.square(residuals / scale), axis=1))
    rat = np.full(len(cpdz), np.nan)
    rat[59:] = cpdz[: len(cpdz) - 59]
    ref = SimpleNamespace(
        scale=scale,
        gepfm_ref_mean=float(np.mean(cpdz)),
        gepfm_ref_scale=float(np.std(cpdz) or 1.0),
        gepfm_smoothing=0.2,
        rat_ref_cov=np.cov(residuals, rowvar=False),
        rat_window=60,
        clean_cpdz=cpdz,
        clean_gepfm=cpdz,
        clean_rat=rat,
    )
    return SimpleNamespace(
        seq_len=12,
        rat_window=60,
        standardizer=standardizer,
        predict_fn=predict,
        ref=ref,
        thresholds={
            "CPDZ residual": 3.0,
            "GEpFM drift": 3.0,
            "RAT covariance": 3.0,
            "Mahalanobis": 3.0,
            "PCA residual": 3.0,
        },
        maha=fit_mahalanobis_detector(clean),
        pca=fit_pca_residual_detector(clean),
        ae=None,
    )


@pytest.fixture
def fitted_encoder() -> HydroJEVEvidenceEncoder:
    rng = np.random.default_rng(42)
    clean = rng.normal(size=(100, 4))
    return HydroJEVEvidenceEncoder.fit(
        clean, ["L_T1", "P_J1", "F_PU1", "S_PU1"], trained=_bundle(clean)
    )


def test_fit_records_reproducibility_metadata(fitted_encoder: HydroJEVEvidenceEncoder) -> None:
    assert fitted_encoder.training_rows == 100
    assert fitted_encoder.gru_epochs == 60
    assert fitted_encoder.ae_epochs == 40
    assert fitted_encoder.seed == 42
    assert len(fitted_encoder.input_sha256) == 64


def test_transform_uses_fixed_causal_grid_and_no_labels(fitted_encoder: HydroJEVEvidenceEncoder) -> None:
    raw = np.random.default_rng(7).normal(size=(90, 4))
    encoded = fitted_encoder.transform(raw)
    np.testing.assert_array_equal(encoded.state_indices, [71, 77, 83, 89])
    assert all(len(values) == len(raw) for values in encoded.scores.values())
    for state in encoded.states:
        text = str(state).lower()
        assert "timestamp" not in text
        assert "dataset" not in text
        assert "row_id" not in text
        assert "att_flag" not in text
        assert state["hydraulic_consistency"]["mass_balance_residual_m3_s"] == "unavailable"


def test_ablation_contracts(fitted_encoder: HydroJEVEvidenceEncoder) -> None:
    raw = np.random.default_rng(8).normal(size=(90, 4))
    temporal = fitted_encoder.transform(raw, ablation="no_temporal").states[0]
    assert "gepfm_drift" not in temporal["detectors"]
    assert "rat_covariance" not in temporal["detectors"]
    assert temporal["temporal_summary"] == {}
    instantaneous = fitted_encoder.transform(raw, ablation="no_instantaneous").states[0]
    assert set(instantaneous["detectors"]) == {"cpdz_residual", "gepfm_drift", "rat_covariance"}
    without_context = fitted_encoder.transform(raw, ablation="no_sensor_context").states[0]
    assert "sensor_context" not in without_context


def test_numeric_features_are_finite_and_same_evidence(fitted_encoder: HydroJEVEvidenceEncoder) -> None:
    raw = np.random.default_rng(9).normal(size=(90, 4))
    encoded = fitted_encoder.transform(raw)
    assert np.isfinite(encoded.numeric_features).all()
    assert encoded.numeric_feature_mask is not None
    assert encoded.numeric_feature_mask.shape == encoded.numeric_features.shape
    assert encoded.numeric_feature_names


def test_rejects_bad_input_and_round_trips(tmp_path: Path, fitted_encoder: HydroJEVEvidenceEncoder) -> None:
    with pytest.raises(ValueError, match="NaN"):
        fitted_encoder.transform(np.array([[np.nan] * 4]))
    with pytest.raises(ValueError, match="width"):
        fitted_encoder.transform(np.zeros((90, 3)))
    raw = np.random.default_rng(10).normal(size=(90, 4))
    path = tmp_path / "encoder.pkl"
    fitted_encoder.save(path)
    loaded = HydroJEVEvidenceEncoder.load(path)
    a = fitted_encoder.transform(raw)
    b = loaded.transform(raw)
    np.testing.assert_array_equal(a.state_indices, b.state_indices)
    np.testing.assert_allclose(a.numeric_features, b.numeric_features)
