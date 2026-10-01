"""Compact two-layer GRU surrogate with leakage-safe preprocessing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from time import perf_counter_ns
from typing import Iterable
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training_values: np.ndarray) -> "Standardizer":
        values = np.asarray(training_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("training_values must be a non-empty 2D array")
        if not np.isfinite(values).all():
            raise ValueError("training_values contain NaN or infinity")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale <= np.finfo(float).eps, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError("feature dimension does not match fitted standardizer")
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError("feature dimension does not match fitted standardizer")
        return array * self.scale + self.mean


@dataclass(frozen=True)
class DeviceSelection:
    device: torch.device
    reason: str


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    model: "GRUSurrogate"
    device: DeviceSelection
    history: tuple[EpochMetrics, ...]
    best_epoch: int


class GRUSurrogate(nn.Module):
    """Rodrigue v1 skeleton adapted to WDN boundary-state regression."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        hidden_size: int = 20,
        num_layers: int = 2,
        dense_size: int = 60,
    ) -> None:
        super().__init__()
        if min(input_size, output_size, hidden_size, num_layers, dense_size) <= 0:
            raise ValueError("all architecture dimensions must be positive")
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, dense_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(dense_size, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("GRU input must have shape [batch, sequence, features]")
        sequence, _ = self.gru(values)
        return self.output(self.activation(self.dense(sequence[:, -1, :])))


class DeviceAwareAdam(torch.optim.Adam):
    """Keep CPU Adam usable when an unusable CUDA device is still enumerated.

    PyTorch 2.14's generic optimizer health check queries the current accelerator
    even when every parameter is on CPU.  On this workstation that initializes an
    unsupported sm_120 device.  Skipping only that graph-capture check for an
    all-CPU optimizer preserves the ordinary Adam update implementation.
    """

    def _accelerator_graph_capture_health_check(self) -> None:
        parameters = (
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        )
        if all(parameter.device.type == "cpu" for parameter in parameters):
            return
        super()._accelerator_graph_capture_health_check()


def select_torch_device(prefer_cuda: bool = True) -> DeviceSelection:
    """Select CUDA only after an actual kernel and synchronization succeed."""

    if not prefer_cuda:
        return DeviceSelection(torch.device("cpu"), "CUDA preference disabled")
    if not torch.cuda.is_available():
        return DeviceSelection(torch.device("cpu"), "torch.cuda.is_available() is false")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            device = torch.device("cuda")
            left = torch.ones((32, 32), device=device)
            right = torch.ones((32, 32), device=device)
            product = left @ right
            torch.cuda.synchronize(device)
            if float(product[0, 0].item()) != 32.0:
                raise RuntimeError("CUDA matrix multiplication returned an invalid probe value")
        return DeviceSelection(device, "CUDA matrix multiplication probe succeeded")
    except Exception as exc:  # Runtime/driver incompatibilities vary by PyTorch build.
        reason = f"CUDA execution probe failed; using CPU: {type(exc).__name__}: {exc}"
        return DeviceSelection(torch.device("cpu"), reason)


def build_sequence_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological windows whose target is the final window timestamp."""

    input_values = np.asarray(inputs, dtype=np.float32)
    target_values = np.asarray(targets, dtype=np.float32)
    if input_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("inputs and targets must both be 2D")
    if len(input_values) != len(target_values):
        raise ValueError("inputs and targets must have equal observation counts")
    if sequence_length < 1 or sequence_length > len(input_values):
        raise ValueError("sequence_length must fit within the observation count")
    count = len(input_values) - sequence_length + 1
    windows = np.stack(
        [input_values[index : index + sequence_length] for index in range(count)], axis=0
    )
    aligned_targets = target_values[sequence_length - 1 :].copy()
    return windows, aligned_targets


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("training data contain NaN or infinity")
    return torch.from_numpy(array)


def train_surrogate(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    *,
    hidden_size: int = 20,
    num_layers: int = 2,
    dense_size: int = 60,
    learning_rate: float = 1e-3,
    minimum_learning_rate: float = 1e-6,
    max_epochs: int = 100,
    batch_size: int = 128,
    early_stopping_patience: int = 10,
    gradient_clip_norm: float = 1.0,
    seed: int = 42,
    device: DeviceSelection | None = None,
) -> TrainingResult:
    """Train a compact surrogate and restore the best validation checkpoint."""

    train_x = _as_float_tensor(train_inputs)
    train_y = _as_float_tensor(train_targets)
    val_x = _as_float_tensor(validation_inputs)
    val_y = _as_float_tensor(validation_targets)
    if train_x.ndim != 3 or val_x.ndim != 3 or train_y.ndim != 2 or val_y.ndim != 2:
        raise ValueError("inputs must be 3D sequences and targets must be 2D")
    if len(train_x) != len(train_y) or len(val_x) != len(val_y):
        raise ValueError("input and target sample counts must match")
    if train_x.shape[2] != val_x.shape[2] or train_y.shape[1] != val_y.shape[1]:
        raise ValueError("training and validation feature dimensions must match")
    if min(max_epochs, batch_size, early_stopping_patience) < 1:
        raise ValueError("epoch, batch, and patience settings must be positive")

    np.random.seed(seed)
    # Seed the CPU generator directly. ``torch.manual_seed`` also initializes
    # CUDA and emits compatibility warnings even when the selected device is CPU.
    torch.random.default_generator.manual_seed(seed)
    selection = device or select_torch_device()
    model = GRUSurrogate(
        input_size=int(train_x.shape[2]),
        output_size=int(train_y.shape[1]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dense_size=dense_size,
    ).to(selection.device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        generator=generator,
    )
    optimizer = DeviceAwareAdam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[EpochMetrics] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0

    validation_x_device = val_x.to(selection.device)
    validation_y_device = val_y.to(selection.device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.to(selection.device))
            loss = criterion(prediction, batch_y.to(selection.device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                criterion(model(validation_x_device), validation_y_device).detach().cpu()
            )
        current_lr = max(
            minimum_learning_rate,
            learning_rate * (0.1 ** ((epoch - 1) // 10)),
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(np.mean(batch_losses)),
                validation_loss=validation_loss,
                learning_rate=current_lr,
            )
        )
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        device=selection,
        history=tuple(history),
        best_epoch=best_epoch,
    )


def predict_surrogate(
    model: GRUSurrogate,
    inputs: np.ndarray,
    device: DeviceSelection,
) -> np.ndarray:
    values = _as_float_tensor(inputs)
    model.eval()
    with torch.inference_mode():
        prediction = model(values.to(device.device)).detach().cpu().numpy()
    return prediction


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_inference_latency(
    model: GRUSurrogate,
    sample: np.ndarray,
    device: DeviceSelection,
    *,
    iterations: int = 100,
    warmup: int = 10,
) -> dict[str, float | int]:
    """Measure warmup-excluded inference latency percentiles in milliseconds."""

    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    values = _as_float_tensor(sample).to(device.device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(values)
        _synchronize_if_needed(device.device)
        timings: list[float] = []
        for _ in range(iterations):
            start = perf_counter_ns()
            model(values)
            _synchronize_if_needed(device.device)
            timings.append((perf_counter_ns() - start) / 1_000_000.0)
    return {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "samples": iterations,
    }
