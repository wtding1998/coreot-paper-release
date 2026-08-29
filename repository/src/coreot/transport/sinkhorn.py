from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SparseUOTResult:
    source_index: np.ndarray
    target_index: np.ndarray
    coupling: np.ndarray
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    n_iter: int
    converged: bool
    terminal_scaling_change: float
    fixed_point_residual: float
    kernel_floor_active: bool
    denominator_floor_active: bool
    nonfinite_detected: bool
    iteration_cap_reached: bool


@dataclass(frozen=True)
class SparseBalancedOTResult:
    source_index: np.ndarray
    target_index: np.ndarray
    coupling: np.ndarray
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    n_iter: int
    converged: bool


@dataclass(frozen=True)
class DenseBalancedOTResult:
    coupling: np.ndarray
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    n_iter: int
    converged: bool


def solve_dense_balanced_sinkhorn(
    *,
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
    numerical_floor: float = 1.0e-300,
) -> DenseBalancedOTResult:
    """Solve entropy-regularized balanced OT on full source-target support."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if cost.shape != (len(source_mass), len(target_mass)):
        raise ValueError("cost shape must match source_mass and target_mass lengths")

    kernel = np.exp(-cost.astype(float) / float(epsilon))
    kernel = np.maximum(kernel, numerical_floor)
    source_scale = np.ones(len(source_mass), dtype=float)
    target_scale = np.ones(len(target_mass), dtype=float)
    converged = False

    for iteration in range(1, max_iter + 1):
        source_scale = source_mass / np.maximum(kernel @ target_scale, numerical_floor)
        target_scale = target_mass / np.maximum(kernel.T @ source_scale, numerical_floor)

        if iteration % 10 == 0 or iteration == 1:
            source_marginal = source_scale * (kernel @ target_scale)
            target_marginal = target_scale * (kernel.T @ source_scale)
            source_error = np.max(np.abs(source_marginal - source_mass))
            target_error = np.max(np.abs(target_marginal - target_mass))
            if max(source_error, target_error) <= tol:
                converged = True
                break
    else:
        iteration = max_iter

    coupling = (source_scale[:, None] * kernel) * target_scale[None, :]
    source_marginal = coupling.sum(axis=1)
    target_marginal = coupling.sum(axis=0)
    return DenseBalancedOTResult(
        coupling=coupling,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        n_iter=iteration,
        converged=converged,
    )


def solve_sparse_balanced_sinkhorn(
    *,
    source_index: np.ndarray,
    target_index: np.ndarray,
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
    numerical_floor: float = 1.0e-300,
) -> SparseBalancedOTResult:
    """Solve entropy-regularized balanced OT on a fixed sparse support."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if len(source_index) != len(target_index) or len(source_index) != len(cost):
        raise ValueError("source_index, target_index, and cost must have the same length")

    kernel = np.exp(-cost.astype(float) / float(epsilon))
    kernel = np.maximum(kernel, numerical_floor)
    source_scale = np.ones(len(source_mass), dtype=float)
    target_scale = np.ones(len(target_mass), dtype=float)
    converged = False

    for iteration in range(1, max_iter + 1):
        previous_source_scale = source_scale.copy()
        previous_target_scale = target_scale.copy()

        source_denominator = np.bincount(
            source_index,
            weights=kernel * target_scale[target_index],
            minlength=len(source_mass),
        )
        source_scale = source_mass / np.maximum(source_denominator, numerical_floor)

        target_denominator = np.bincount(
            target_index,
            weights=kernel * source_scale[source_index],
            minlength=len(target_mass),
        )
        target_scale = target_mass / np.maximum(target_denominator, numerical_floor)

        source_delta = np.max(np.abs(source_scale - previous_source_scale))
        target_delta = np.max(np.abs(target_scale - previous_target_scale))
        if max(source_delta, target_delta) <= tol:
            converged = True
            break
    else:
        iteration = max_iter

    coupling = source_scale[source_index] * kernel * target_scale[target_index]
    source_marginal = np.bincount(source_index, weights=coupling, minlength=len(source_mass))
    target_marginal = np.bincount(target_index, weights=coupling, minlength=len(target_mass))
    return SparseBalancedOTResult(
        source_index=source_index,
        target_index=target_index,
        coupling=coupling,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        n_iter=iteration,
        converged=converged,
    )


def solve_sparse_unbalanced_sinkhorn(
    *,
    source_index: np.ndarray,
    target_index: np.ndarray,
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    epsilon: float,
    tau_source: np.ndarray,
    tau_target: np.ndarray,
    max_iter: int,
    tol: float,
    numerical_floor: float = 1.0e-300,
) -> SparseUOTResult:
    """Solve entropy-regularized unbalanced OT on a fixed sparse support."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if len(source_index) != len(target_index) or len(source_index) != len(cost):
        raise ValueError("source_index, target_index, and cost must have the same length")
    if np.any(tau_source <= 0) or np.any(tau_target <= 0):
        raise ValueError("tau_source and tau_target must be positive")

    raw_kernel = np.exp(-cost.astype(float) / float(epsilon))
    kernel_floor_active = bool(np.any(raw_kernel < numerical_floor))
    kernel = raw_kernel
    kernel = np.maximum(kernel, numerical_floor)
    source_scale = np.ones(len(source_mass), dtype=float)
    target_scale = np.ones(len(target_mass), dtype=float)
    source_power = tau_source / (tau_source + epsilon)
    target_power = tau_target / (tau_target + epsilon)
    converged = False
    denominator_floor_active = False
    nonfinite_detected = False
    terminal_scaling_change = float("inf")

    for iteration in range(1, max_iter + 1):
        previous_source_scale = source_scale.copy()
        previous_target_scale = target_scale.copy()

        source_denominator = np.bincount(
            source_index,
            weights=kernel * target_scale[target_index],
            minlength=len(source_mass),
        )
        denominator_floor_active = denominator_floor_active or bool(
            np.any(source_denominator < numerical_floor)
        )
        source_scale = np.power(
            source_mass / np.maximum(source_denominator, numerical_floor),
            source_power,
        )

        target_denominator = np.bincount(
            target_index,
            weights=kernel * source_scale[source_index],
            minlength=len(target_mass),
        )
        denominator_floor_active = denominator_floor_active or bool(
            np.any(target_denominator < numerical_floor)
        )
        target_scale = np.power(
            target_mass / np.maximum(target_denominator, numerical_floor),
            target_power,
        )

        if not (np.isfinite(source_scale).all() and np.isfinite(target_scale).all()):
            nonfinite_detected = True
            break
        source_delta = float(np.max(np.abs(source_scale - previous_source_scale)))
        target_delta = float(np.max(np.abs(target_scale - previous_target_scale)))
        terminal_scaling_change = max(source_delta, target_delta)
        if terminal_scaling_change <= tol:
            converged = True
            break
    else:
        iteration = max_iter

    check_source_denominator = np.bincount(
        source_index,
        weights=kernel * target_scale[target_index],
        minlength=len(source_mass),
    )
    check_source_scale = np.power(
        source_mass / np.maximum(check_source_denominator, numerical_floor),
        source_power,
    )
    check_target_denominator = np.bincount(
        target_index,
        weights=kernel * check_source_scale[source_index],
        minlength=len(target_mass),
    )
    check_target_scale = np.power(
        target_mass / np.maximum(check_target_denominator, numerical_floor),
        target_power,
    )
    if np.isfinite(check_source_scale).all() and np.isfinite(check_target_scale).all():
        source_residual = np.abs(check_source_scale - source_scale) / np.maximum(
            1.0, np.abs(source_scale)
        )
        target_residual = np.abs(check_target_scale - target_scale) / np.maximum(
            1.0, np.abs(target_scale)
        )
        fixed_point_residual = max(
            float(np.max(source_residual)),
            float(np.max(target_residual)),
        )
    else:
        fixed_point_residual = float("inf")
        nonfinite_detected = True

    coupling = source_scale[source_index] * kernel * target_scale[target_index]
    source_marginal = np.bincount(source_index, weights=coupling, minlength=len(source_mass))
    target_marginal = np.bincount(target_index, weights=coupling, minlength=len(target_mass))
    return SparseUOTResult(
        source_index=source_index,
        target_index=target_index,
        coupling=coupling,
        source_marginal=source_marginal,
        target_marginal=target_marginal,
        n_iter=iteration,
        converged=converged,
        terminal_scaling_change=terminal_scaling_change,
        fixed_point_residual=fixed_point_residual,
        kernel_floor_active=kernel_floor_active,
        denominator_floor_active=denominator_floor_active,
        nonfinite_detected=nonfinite_detected,
        iteration_cap_reached=bool(iteration == max_iter and not converged),
    )
