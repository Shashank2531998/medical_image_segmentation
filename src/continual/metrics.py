from __future__ import annotations

from typing import Iterable
import numpy as np


def compute_average_performance_matrix(eval_matrix: np.ndarray) -> np.ndarray:
    """Compute Average Performance A_t for each task t from an evaluation matrix.

    eval_matrix is expected shape (T, T) where row t (0-based) contains the
    performance a_{t,i} for tasks i (0-based) after training up to task t.

    Returns array A of shape (T,) where A[t] = (1/(t+1)) * sum_{i=0..t} a_{t,i}.
    Missing entries (np.nan) are ignored when computing the mean.
    """
    eval_matrix = np.asarray(eval_matrix, dtype=float)
    T = eval_matrix.shape[0]
    A = np.empty(T, dtype=float)
    for t in range(T):
        row = eval_matrix[t, : t + 1]
        A[t] = np.nanmean(row)
    return A


def compute_forgetting_matrix(eval_matrix: np.ndarray) -> np.ndarray:
    """Compute forgetting measures F_t for each task t.

    For t>0, for each task i < t:
        f_{t,i} = max_{s in {0..t-1}} a_{s,i} - a_{t,i}
    Then F_t = (1/(t)) * sum_{i=0..t-1} f_{t,i}  (note denominator t since 0-based)

    Returns array F of shape (T,) where F[0] = 0.0.
    """
    eval_matrix = np.asarray(eval_matrix, dtype=float)
    T = eval_matrix.shape[0]
    F = np.zeros(T, dtype=float)
    for t in range(1, T):
        f_vals = []
        for i in range(0, t):
            prev_vals = eval_matrix[:t, i]
            # ignore nans in prev_vals
            if np.all(np.isnan(prev_vals)) or np.isnan(eval_matrix[t, i]):
                # cannot compute forgetting for this task/sample
                continue
            max_prev = np.nanmax(prev_vals)
            f_i = float(max_prev - eval_matrix[t, i])
            f_vals.append(f_i)
        if f_vals:
            F[t] = float(np.mean(f_vals))
        else:
            F[t] = 0.0
    return F


def compute_all_metrics(eval_matrix: Iterable[Iterable[float]]) -> dict:
    """Convenience: compute A_t and F_t for all t and return dict.

    Returns {'A': A_array, 'F': F_array}
    """
    mat = np.asarray(eval_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("eval_matrix must be a square 2D array (T x T)")
    A = compute_average_performance_matrix(mat)
    F = compute_forgetting_matrix(mat)
    return {"A": A, "F": F}
