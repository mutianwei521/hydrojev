"""Erba-style expanding reconstruction detector and constrained concealment attack."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hydrojev.physics.surrogate_models import (
    DeviceAwareAdam,
    DeviceSelection,
    select_torch_device,
)


@dataclass(frozen=True)
class FeatureRangeScaler:
    minimum: np.ndarray
    maximum: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, training_values: np.ndarray) -> "FeatureRangeScaler":
        values = np.asarray(training_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("training_values must be a non-empty 2D array")
        if not np.isfinite(values).all():
            raise ValueError("training_values contain NaN or infinity")
        minimum = values.min(axis=0)
        maximum = values.max(axis=0)
        raw_span = maximum - minimum
        span = np.where(raw_span <= np.finfo(float).eps, 1.0, raw_span)
        return cls(minimum=minimum, maximum=maximum, span=span)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.minimum.shape[0]:
            raise ValueError("feature dimension does not match fitted scaler")
        return (array - self.minimum) / self.span

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.minimum.shape[0]:
            raise ValueError("feature dimension does not match fitted scaler")
        return array * self.span + self.minimum


class ConcealmentAutoencoder(nn.Module):
    """The ACSAC 2020 black-box n→2n→4n→2n→n architecture."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.Sigmoid(),
            nn.Linear(2 * input_dim, 4 * input_dim),
            nn.Sigmoid(),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.Sigmoid(),
            nn.Linear(2 * input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("autoencoder input must have shape [batch, features]")
        return self.network(values)


@dataclass(frozen=True)
class ReconstructionDetector:
    model: ConcealmentAutoencoder
    scaler: FeatureRangeScaler
    threshold: float
    device: DeviceSelection

    def _scaled_reconstruction(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        array = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if not np.isfinite(array).all():
            raise ValueError("detector input contains NaN or infinity")
        scaled = self.scaler.transform(array).astype(np.float32)
        tensor = torch.from_numpy(scaled).to(self.device.device)
        self.model.eval()
        with torch.inference_mode():
            reconstructed = self.model(tensor).detach().cpu().numpy()
        return scaled, reconstructed

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        scaled, reconstructed = self._scaled_reconstruction(values)
        return np.mean(np.square(scaled - reconstructed), axis=1)

    def feature_errors(self, value: np.ndarray) -> np.ndarray:
        scaled, reconstructed = self._scaled_reconstruction(np.asarray(value))
        if len(scaled) != 1:
            raise ValueError("feature_errors accepts one observation")
        return np.square(scaled[0] - reconstructed[0])

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        _, reconstructed = self._scaled_reconstruction(values)
        return self.scaler.inverse_transform(reconstructed)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (self.score_samples(values) >= self.threshold).astype(np.int8)


@dataclass(frozen=True)
class AutoencoderEpoch:
    epoch: int
    train_loss: float
    validation_loss: float
    learning_rate: float


@dataclass(frozen=True)
class DetectorTrainingResult:
    detector: ReconstructionDetector
    history: tuple[AutoencoderEpoch, ...]
    best_epoch: int


def _tensor(values: np.ndarray) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("values contain NaN or infinity")
    return torch.from_numpy(array)


def train_reconstruction_detector(
    training_values: np.ndarray,
    validation_values: np.ndarray,
    *,
    threshold_quantile: float = 0.995,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    batch_size: int = 128,
    early_stopping_patience: int = 10,
    seed: int = 42,
    device: DeviceSelection | None = None,
) -> DetectorTrainingResult:
    """Fit preprocessing on normal training data and calibrate its score threshold."""

    training = np.asarray(training_values, dtype=np.float64)
    validation = np.asarray(validation_values, dtype=np.float64)
    if training.ndim != 2 or validation.ndim != 2 or training.shape[1] != validation.shape[1]:
        raise ValueError("training and validation must be 2D with equal feature counts")
    if min(len(training), len(validation)) < 1:
        raise ValueError("training and validation cannot be empty")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be in (0, 1)")
    if min(max_epochs, batch_size, early_stopping_patience) < 1:
        raise ValueError("epoch, batch, and patience settings must be positive")

    scaler = FeatureRangeScaler.fit(training)
    train_tensor = _tensor(scaler.transform(training))
    validation_tensor = _tensor(scaler.transform(validation))
    selection = device or select_torch_device()
    torch.random.default_generator.manual_seed(seed)
    model = ConcealmentAutoencoder(training.shape[1]).to(selection.device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(batch_size, len(train_tensor)),
        shuffle=True,
        generator=generator,
    )
    optimizer = DeviceAwareAdam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, early_stopping_patience // 3)
    )
    criterion = nn.MSELoss()
    history: list[AutoencoderEpoch] = []
    best_loss = math.inf
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale = 0
    validation_device = validation_tensor.to(selection.device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for (batch,) in loader:
            batch = batch.to(selection.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                criterion(model(validation_device), validation_device).detach().cpu()
            )
        scheduler.step(validation_loss)
        history.append(
            AutoencoderEpoch(
                epoch=epoch,
                train_loss=float(np.mean(losses)),
                validation_loss=validation_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
        )
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= early_stopping_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    provisional = ReconstructionDetector(model, scaler, math.inf, selection)
    threshold = float(np.quantile(provisional.score_samples(training), threshold_quantile))
    detector = ReconstructionDetector(model, scaler, threshold, selection)
    return DetectorTrainingResult(detector, tuple(history), best_epoch)


@dataclass(frozen=True)
class ActuatorFlowRule:
    status_index: int
    flow_indices: tuple[int, ...]
    closed_status: float = 0.0
    closed_flow_value: float = 0.0


def apply_actuator_rules(
    candidate: np.ndarray,
    *,
    original: np.ndarray,
    discrete_features: Mapping[int, Sequence[float]] | None = None,
    actuator_rules: Sequence[ActuatorFlowRule] = (),
    modifiable_mask: np.ndarray,
    max_modified_features: int,
) -> np.ndarray | None:
    """Snap discrete states and reject postprocessing that exceeds the true budget."""

    processed = np.asarray(candidate, dtype=float).copy()
    baseline = np.asarray(original, dtype=float)
    mask = np.asarray(modifiable_mask, dtype=bool)
    if processed.ndim != 1 or processed.shape != baseline.shape or mask.shape != baseline.shape:
        raise ValueError("candidate, original, and modifiable_mask must share one-dimensional shape")
    for index, allowed_values in (discrete_features or {}).items():
        allowed = np.asarray(tuple(allowed_values), dtype=float)
        if allowed.size == 0:
            raise ValueError(f"discrete feature {index} has no allowed values")
        processed[index] = allowed[np.argmin(np.abs(allowed - processed[index]))]
    for rule in actuator_rules:
        if np.isclose(processed[rule.status_index], rule.closed_status):
            for flow_index in rule.flow_indices:
                processed[flow_index] = rule.closed_flow_value
    changed = ~np.isclose(processed, baseline, rtol=1e-9, atol=1e-12)
    if np.any(changed & ~mask) or int(changed.sum()) > max_modified_features:
        return None
    return processed


class DetectorLike(Protocol):
    scaler: FeatureRangeScaler
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray: ...

    def feature_errors(self, value: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ConcealmentAttackResult:
    original: np.ndarray
    adversarial: np.ndarray
    original_score: float
    adversarial_score: float
    threshold: float
    changed_indices: tuple[int, ...]
    query_count: int
    converged: bool
    termination_reason: str


def coordinate_descent_concealment(
    detector: DetectorLike,
    sample: np.ndarray,
    *,
    modifiable_mask: np.ndarray,
    max_modified_features: int = 4,
    candidate_values: int = 21,
    budget: int = 200,
    patience: int = 15,
    discrete_features: Mapping[int, Sequence[float]] | None = None,
    actuator_rules: Sequence[ActuatorFlowRule] = (),
) -> ConcealmentAttackResult:
    """Minimize reconstruction error with Erba-style constrained coordinate search."""

    original = np.asarray(sample, dtype=float)
    mask = np.asarray(modifiable_mask, dtype=bool)
    if original.ndim != 1 or mask.shape != original.shape:
        raise ValueError("sample and modifiable_mask must share one-dimensional shape")
    if not np.isfinite(original).all():
        raise ValueError("sample contains NaN or infinity")
    if not 1 <= max_modified_features <= 4:
        raise ValueError("max_modified_features must allow at most 4 changed measurements")
    if candidate_values < 2 or budget < 1 or patience < 1:
        raise ValueError("candidate_values, budget, and patience must be positive")
    if detector.scaler.minimum.shape != original.shape:
        raise ValueError("detector scaler dimension does not match sample")

    current = original.copy()
    original_score = float(detector.score_samples(current[None, :])[0])
    current_score = original_score
    query_count = 0
    stale = 0
    reason = "patience_exhausted"

    while True:
        if current_score < detector.threshold:
            reason = "threshold_reached"
            break
        if query_count >= budget:
            reason = "budget_exhausted"
            break
        changed = ~np.isclose(current, original, rtol=1e-9, atol=1e-12)
        eligible = mask.copy()
        if int(changed.sum()) >= max_modified_features:
            eligible &= changed
        if not eligible.any():
            reason = "no_modifiable_coordinate"
            break
        errors = np.asarray(detector.feature_errors(current), dtype=float)
        if errors.shape != current.shape:
            raise ValueError("detector feature_errors returned an incompatible shape")
        ranked = np.argsort(-errors, kind="stable")
        coordinate = next((int(index) for index in ranked if eligible[index]), None)
        if coordinate is None:
            reason = "no_modifiable_coordinate"
            break

        if discrete_features and coordinate in discrete_features:
            candidates = np.asarray(tuple(discrete_features[coordinate]), dtype=float)
        else:
            lower = float(detector.scaler.minimum[coordinate])
            upper = float(detector.scaler.maximum[coordinate])
            candidates = np.linspace(lower, upper, candidate_values)
        best = current
        best_score = current_score
        for candidate_value in candidates:
            if query_count >= budget:
                break
            trial = current.copy()
            trial[coordinate] = candidate_value
            trial = apply_actuator_rules(
                trial,
                original=original,
                discrete_features=discrete_features,
                actuator_rules=actuator_rules,
                modifiable_mask=mask,
                max_modified_features=max_modified_features,
            )
            if trial is None or np.allclose(trial, current, rtol=1e-9, atol=1e-12):
                continue
            score = float(detector.score_samples(trial[None, :])[0])
            query_count += 1
            if score < best_score - 1e-12:
                best = trial
                best_score = score
        if best_score < current_score - 1e-12:
            current = best.copy()
            current_score = best_score
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                reason = "patience_exhausted"
                break

    changed_indices = tuple(
        int(index)
        for index in np.flatnonzero(
            ~np.isclose(current, original, rtol=1e-9, atol=1e-12)
        )
    )
    return ConcealmentAttackResult(
        original=original.copy(),
        adversarial=current.copy(),
        original_score=original_score,
        adversarial_score=current_score,
        threshold=float(detector.threshold),
        changed_indices=changed_indices,
        query_count=query_count,
        converged=current_score < detector.threshold,
        termination_reason=reason,
    )

