from __future__ import annotations

import lap
import numpy as np
from scipy.spatial.distance import cdist


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    _cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matched_a = np.where(x >= 0)[0]
    if matched_a.size:
        matches = np.stack([matched_a, x[matched_a]], axis=1)
    else:
        matches = np.empty((0, 2), dtype=int)
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    return matches, unmatched_a, unmatched_b


def _points_matrix(tracks):
    """Stack track/detection xy without per-item .copy().

    ``tracks`` may be a list of STrack/Detection, or an (N, 2) ndarray.
    """
    if isinstance(tracks, np.ndarray):
        return np.asarray(tracks, dtype=np.float64)
    n = len(tracks)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = np.empty((n, 2), dtype=np.float64)
    for i, t in enumerate(tracks):
        p = t.mean[:2] if getattr(t, "mean", None) is not None else t._point
        out[i, 0] = p[0]
        out[i, 1] = p[1]
    return out


def euclidean_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    return cdist(_points_matrix(atracks), _points_matrix(btracks), metric="euclidean")


def maha_distance(atracks, btracks, kalman_filter_instance, metric="maha"):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    bpoints = _points_matrix(btracks)
    if metric == "euclidean":
        return cdist(_points_matrix(atracks), bpoints, metric="euclidean")

    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    for i, track in enumerate(atracks):
        cost_matrix[i, :] = kalman_filter_instance.gating_distance(
            track.mean, track.covariance, bpoints, metric="maha"
        )
    return cost_matrix
