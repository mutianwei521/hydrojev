import numpy as np
import pytest
import torch

from hydrojev.physics.surrogate_models import (
    GRUSurrogate,
    Standardizer,
    build_sequence_dataset,
    measure_inference_latency,
    predict_surrogate,
    select_torch_device,
    train_surrogate,
)


def test_gru_surrogate_has_paper_v1_compact_architecture() -> None:
    model = GRUSurrogate(input_size=3, output_size=2)

    assert model.gru.num_layers == 2
    assert model.gru.hidden_size == 20
    assert model.dense.in_features == 20
    assert model.dense.out_features == 60
    assert model.output.out_features == 2
    output = model(torch.zeros(4, 12, 3))
    assert output.shape == (4, 2)


def test_standardizer_is_fit_only_from_training_values() -> None:
    training = np.array([[0.0, 10.0], [2.0, 14.0]])
    validation = np.array([[100.0, 200.0]])

    scaler = Standardizer.fit(training)

    np.testing.assert_allclose(scaler.mean, [1.0, 12.0])
    np.testing.assert_allclose(scaler.transform(training).mean(axis=0), [0.0, 0.0])
    assert scaler.transform(validation)[0, 0] == pytest.approx(99.0)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(training)), training)


def test_sequence_builder_keeps_chronological_targets() -> None:
    inputs = np.arange(20, dtype=float).reshape(10, 2)
    targets = np.arange(10, dtype=float).reshape(10, 1)

    sequence_inputs, sequence_targets = build_sequence_dataset(
        inputs, targets, sequence_length=3
    )

    assert sequence_inputs.shape == (8, 3, 2)
    assert sequence_targets[:, 0].tolist() == list(range(2, 10))
    np.testing.assert_array_equal(sequence_inputs[0], inputs[:3])


def test_device_probe_can_be_forced_to_cpu() -> None:
    selection = select_torch_device(prefer_cuda=False)

    assert selection.device.type == "cpu"
    assert "disabled" in selection.reason.lower()


def test_small_training_and_inference_pipeline_is_executable() -> None:
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(36, 3)).astype(np.float32)
    targets = np.column_stack((inputs[:, 0] + inputs[:, 1], inputs[:, 2])).astype(
        np.float32
    )
    train_x, train_y = build_sequence_dataset(inputs[:28], targets[:28], 4)
    val_x, val_y = build_sequence_dataset(inputs[28:], targets[28:], 4)

    result = train_surrogate(
        train_x,
        train_y,
        val_x,
        val_y,
        max_epochs=2,
        batch_size=8,
        seed=11,
        device=select_torch_device(prefer_cuda=False),
    )
    predictions = predict_surrogate(result.model, val_x[:2], result.device)
    latency = measure_inference_latency(
        result.model, val_x[:1], result.device, iterations=3, warmup=1
    )

    assert len(result.history) >= 1
    assert predictions.shape == (2, 2)
    assert set(latency) == {"p50_ms", "p95_ms", "p99_ms", "samples"}
    assert latency["p99_ms"] >= 0.0

