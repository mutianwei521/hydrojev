from dataclasses import dataclass

import numpy as np
import pytest
import torch

from hydrojev.attacks.concealment_ae import (
    ActuatorFlowRule,
    ConcealmentAutoencoder,
    FeatureRangeScaler,
    apply_actuator_rules,
    coordinate_descent_concealment,
    train_reconstruction_detector,
)
from hydrojev.physics.surrogate_models import select_torch_device


def test_concealment_autoencoder_uses_expanding_erba_architecture() -> None:
    model = ConcealmentAutoencoder(input_dim=5)
    widths = [
        module.out_features
        for module in model.network
        if isinstance(module, torch.nn.Linear)
    ]

    assert widths == [10, 20, 10, 5]
    assert sum(isinstance(module, torch.nn.Sigmoid) for module in model.network) == 4
    assert model(torch.zeros(2, 5)).shape == (2, 5)


def test_feature_range_scaler_fits_training_only_and_handles_constant_columns() -> None:
    training = np.array([[0.0, 5.0], [2.0, 5.0]])
    validation = np.array([[100.0, 5.0]])

    scaler = FeatureRangeScaler.fit(training)

    np.testing.assert_allclose(scaler.minimum, [0.0, 5.0])
    np.testing.assert_allclose(scaler.maximum, [2.0, 5.0])
    np.testing.assert_allclose(scaler.transform(training), [[0.0, 0.0], [1.0, 0.0]])
    assert scaler.transform(validation)[0, 0] == pytest.approx(50.0)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(training)), training)


def test_training_sets_threshold_from_training_score_quantile() -> None:
    rng = np.random.default_rng(3)
    training = rng.uniform(0.0, 1.0, size=(24, 3)).astype(np.float32)
    validation = rng.uniform(0.0, 1.0, size=(8, 3)).astype(np.float32)

    result = train_reconstruction_detector(
        training,
        validation,
        max_epochs=2,
        batch_size=8,
        threshold_quantile=0.75,
        device=select_torch_device(prefer_cuda=False),
        seed=5,
    )
    scores = result.detector.score_samples(training)

    assert result.detector.threshold == pytest.approx(np.quantile(scores, 0.75))
    assert result.best_epoch >= 1
    assert len(result.history) >= 1


@dataclass
class _QuadraticDetector:
    scaler: FeatureRangeScaler
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        array = np.atleast_2d(np.asarray(values, dtype=float))
        return np.mean(array**2, axis=1)

    def feature_errors(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=float) ** 2


def test_coordinate_descent_reduces_score_with_at_most_k_changed_features() -> None:
    detector = _QuadraticDetector(
        scaler=FeatureRangeScaler(
            minimum=np.zeros(3), maximum=np.ones(3), span=np.ones(3)
        ),
        threshold=0.01,
    )

    result = coordinate_descent_concealment(
        detector,
        np.array([1.0, 0.8, 0.7]),
        modifiable_mask=np.array([True, True, True]),
        max_modified_features=1,
        candidate_values=3,
        budget=20,
        patience=3,
    )

    assert result.adversarial_score < result.original_score
    assert result.changed_indices == (0,)
    assert np.count_nonzero(result.adversarial != result.original) <= 1
    assert result.query_count <= 20


def test_coordinate_descent_rejects_more_than_four_modified_features() -> None:
    detector = _QuadraticDetector(
        scaler=FeatureRangeScaler(
            minimum=np.zeros(5), maximum=np.ones(5), span=np.ones(5)
        ),
        threshold=0.01,
    )

    with pytest.raises(ValueError, match="at most 4"):
        coordinate_descent_concealment(
            detector,
            np.ones(5),
            modifiable_mask=np.ones(5, dtype=bool),
            max_modified_features=5,
        )


def test_discrete_status_and_closed_flow_rule_count_all_changed_coordinates() -> None:
    sample = np.array([1.0, 12.5, 3.0])
    rule = ActuatorFlowRule(status_index=0, flow_indices=(1,), closed_status=0.0)

    processed = apply_actuator_rules(
        np.array([0.1, 12.5, 3.0]),
        original=sample,
        discrete_features={0: (0.0, 1.0)},
        actuator_rules=(rule,),
        modifiable_mask=np.array([True, True, False]),
        max_modified_features=2,
    )

    assert processed is not None
    np.testing.assert_allclose(processed, [0.0, 0.0, 3.0])
    assert (
        apply_actuator_rules(
            np.array([0.1, 12.5, 3.0]),
            original=sample,
            discrete_features={0: (0.0, 1.0)},
            actuator_rules=(rule,),
            modifiable_mask=np.array([True, True, False]),
            max_modified_features=1,
        )
        is None
    )

