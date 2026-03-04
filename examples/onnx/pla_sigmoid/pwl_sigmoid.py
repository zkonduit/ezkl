"""
Piecewise Linear Approximation (PLA) for Sigmoid — replaces Lookup Table in ZK circuits.

- Uses only multiplications, additions, and comparisons (no exp, no lookup).
- Enables circuits to use copy/arithmetic constraints instead of lookup tables.
- Target accuracy: >= 99.5% (max relative error <= 0.5%).
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

DEFAULT_X_MIN = -2.0
DEFAULT_X_MAX = 2.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def fit_pwl_sigmoid(
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    n_segments: int = 16,
    n_samples_per_seg: int = 200,
    target_max_rel_error: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit piecewise linear segments to sigmoid; returns breakpoints, slopes, intercepts."""
    breakpoints = np.linspace(x_min, x_max, n_segments + 1)
    slopes = np.zeros(n_segments)
    intercepts = np.zeros(n_segments)

    for i in range(n_segments):
        left, right = breakpoints[i], breakpoints[i + 1]
        xs = np.linspace(left, right, n_samples_per_seg)
        ys = sigmoid(xs)
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        (a, b), _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
        slopes[i] = a
        intercepts[i] = b

    x_check = np.linspace(x_min, x_max, 2000)
    y_true = sigmoid(x_check)
    y_pwl = np.zeros_like(x_check)
    for i in range(n_segments):
        mask = (x_check >= breakpoints[i]) & (x_check < breakpoints[i + 1])
        if i == n_segments - 1:
            mask = (x_check >= breakpoints[i]) & (x_check <= breakpoints[i + 1])
        y_pwl[mask] = slopes[i] * x_check[mask] + intercepts[i]
    denom = np.maximum(np.abs(y_true), 1e-8)
    rel_err = np.abs(y_pwl - y_true) / denom
    max_rel = float(np.max(rel_err))

    if max_rel > target_max_rel_error and n_segments < 64:
        return fit_pwl_sigmoid(
            x_min, x_max, n_segments=min(n_segments + 8, 64),
            n_samples_per_seg=n_samples_per_seg,
            target_max_rel_error=target_max_rel_error,
        )
    return breakpoints, slopes, intercepts


def pwl_sigmoid_numpy(x, breakpoints, slopes, intercepts):
    """NumPy PWL sigmoid for verification."""
    out = np.zeros_like(x, dtype=np.float64)
    n_segments = len(slopes)
    for i in range(n_segments):
        left, right = breakpoints[i], breakpoints[i + 1]
        mask = (x >= left) & (x <= right) if i == n_segments - 1 else (x >= left) & (x < right)
        out[mask] = slopes[i] * x[mask] + intercepts[i]
    out[x < breakpoints[0]] = sigmoid(x[x < breakpoints[0]])
    out[x > breakpoints[-1]] = sigmoid(x[x > breakpoints[-1]])
    return out


class PWLSigmoid(nn.Module):
    """Piecewise linear Sigmoid: only mul/add/compare, no lookup. ZK-friendly."""

    def __init__(self, breakpoints: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray):
        super().__init__()
        self.register_buffer("breakpoints", torch.tensor(breakpoints, dtype=torch.float32))
        self.register_buffer("slopes", torch.tensor(slopes, dtype=torch.float32))
        self.register_buffer("intercepts", torch.tensor(intercepts, dtype=torch.float32))
        self.n_segments = len(slopes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.breakpoints[0], self.breakpoints[-1])
        out = torch.zeros_like(x)
        for i in range(self.n_segments):
            left = self.breakpoints[i]
            right = self.breakpoints[i + 1]
            mask = (x >= left) & (x <= right) if i == self.n_segments - 1 else (x >= left) & (x < right)
            out = out + mask.float() * (self.slopes[i] * x + self.intercepts[i])
        return out


def calibrate_and_save(
    x_min: float = DEFAULT_X_MIN,
    x_max: float = DEFAULT_X_MAX,
    target_accuracy: float = 0.995,
    out_path: str = "pwl_params.npz",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate PWL to meet target accuracy and save to npz."""
    target_max_rel = 1.0 - target_accuracy
    for n in [8, 16, 32, 64]:
        try:
            bp, sl, ic = fit_pwl_sigmoid(x_min, x_max, n_segments=n, target_max_rel_error=target_max_rel)
            x_check = np.linspace(x_min, x_max, 2000)
            y_true = sigmoid(x_check)
            y_pwl = pwl_sigmoid_numpy(x_check, bp, sl, ic)
            rel_err = np.abs(y_pwl - y_true) / np.maximum(np.abs(y_true), 1e-8)
            max_rel = float(np.max(rel_err))
            if (1.0 - max_rel) >= target_accuracy:
                np.savez(out_path, breakpoints=bp, slopes=sl, intercepts=ic)
                return bp, sl, ic
        except Exception:
            continue
    raise RuntimeError(f"Could not reach {target_accuracy*100}% accuracy with 8--64 segments")


if __name__ == "__main__":
    calibrate_and_save(target_accuracy=0.995)
