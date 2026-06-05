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
    """
    Compute average forgetting F_t at each time step t.

    Interpretation:
        eval_matrix[i, t] = performance on task i
                            evaluated at checkpoint t

    F_t:
        average over tasks i < t of:
            max_{s < t} a[i, s] - a[i, t]
    """
    eval_matrix = np.asarray(eval_matrix, dtype=float)

    num_tasks, num_checkpoints = eval_matrix.shape

    # assuming square CL setting: T tasks, T+1 checkpoints sometimes
    T = min(num_tasks, num_checkpoints)

    F = np.zeros(T, dtype=float)

    for t in range(1, T):

        f_vals = []

        for i in range(t):

            # history of task i over time (correct axis!)
            history = eval_matrix[i, :t]

            current = eval_matrix[i, t]

            # skip invalid cases
            if np.all(np.isnan(history)) or np.isnan(current):
                continue

            max_past = np.nanmax(history)

            f_vals.append(max_past - current)

        F[t] = np.mean(f_vals) if f_vals else 0.0

    return F


def compute_zscl_transfer(eval_matrix: np.ndarray) -> np.ndarray:
    """
    Compute ZSCL Transfer metric.

    Definition:
        Transfer_t = (1 / (t-1)) * sum_{i=0..t-1} a[i, t]

    Parameters
    ----------
    cl_matrix : np.ndarray
        Shape (num_tasks, num_checkpoints)
        cl_matrix[i, t] = performance on task i at checkpoint t

    Returns
    -------
    np.ndarray
        Shape (num_checkpoints,)
        Transfer score per task step
    """
    eval_matrix = np.asarray(eval_matrix, dtype=float)

    num_tasks, num_checkpoints = eval_matrix.shape

    transfer = np.zeros(num_tasks, dtype=float)

    for t in range(0, num_tasks):
        future_tasks = eval_matrix[t, :t]

        transfer[t] = np.nanmean(future_tasks)

    return transfer


def compute_final_average_performance(cl_matrix: np.ndarray) -> float:
    """
    Average CL performance at the final checkpoint.
    """
    cl_matrix = np.asarray(cl_matrix, dtype=float)

    final_scores = cl_matrix[:, -1]

    return float(np.nanmean(final_scores))


def compute_retention_drop(retention_matrix: np.ndarray):
    retention_matrix = np.asarray(retention_matrix, dtype=float)

    num_tasks, num_checkpoints = retention_matrix.shape

    zs = np.zeros(num_checkpoints, dtype=float)

    for i in range(1, num_checkpoints):
        baseline = retention_matrix[:, 0]

        # difference over all checkpoints
        zs[i] = np.nanmean(baseline - retention_matrix[:, i])

    return zs[1:]


def compute_all_metrics(
    eval_matrix: Iterable[Iterable[float]],
    num_training_tasks: int,
) -> dict:
    """
    Compute all continual-learning and retention metrics.

    Expected matrix layout:

                        Pretrained  AfterT1 ... AfterTN
    CL Task 1
    ...
    CL Task N
    Retention Task(s)

    Parameters
    ----------
    eval_matrix
        Full evaluation matrix.

    num_training_tasks
        Number of continual-learning tasks.

    Returns
    -------
    dict
        {
            "A": np.ndarray,
            "F": np.ndarray,
            "ZS": np.ndarray,
            "A_final": float,
            "retention_drop": np.ndarray
        }
    """
    mat = np.asarray(eval_matrix, dtype=float)

    if mat.ndim != 2:
        raise ValueError(
            "eval_matrix must be a 2D array"
        )

    n_rows, n_cols = mat.shape

    expected_cols = num_training_tasks + 1

    if n_cols != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns "
            f"(pretrained + {num_training_tasks} checkpoints), "
            f"got {n_cols}"
        )

    if n_rows < num_training_tasks:
        raise ValueError(
            "num_training_tasks exceeds number of rows"
        )

    # --------------------------------------------------
    # Split CL and retention rows
    # --------------------------------------------------

    cl_matrix = mat[:num_training_tasks]
    retention_matrix = mat[num_training_tasks:]

    # --------------------------------------------------
    # CL metrics
    # --------------------------------------------------

    # Remove pretrained column for standard CL metrics
    cl_after_training = cl_matrix[:, 1:]

    A = compute_average_performance_matrix(
        cl_after_training
    )

    F = compute_forgetting_matrix(
        cl_after_training
    )

    ZS = compute_zscl_transfer(
        cl_after_training
    )

    A_final = compute_final_average_performance(
        cl_matrix
    )

    retention_drop = compute_retention_drop(
        retention_matrix
    ) if retention_matrix.shape[0] > 0 else np.array([], dtype=float)

    return {
        "A": A,
        "F": F,
        "ZS": ZS,
        "A_final": A_final,
        "retention_drop": retention_drop
    }
