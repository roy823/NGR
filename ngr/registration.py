"""Coordinate-to-pose interfaces with a common physical residual radius."""
import numpy as np

from ._core import bank
from ._rigid import seed_poses
from ._surface import process


def _matches(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or p.shape != q.shape or len(p) < 3:
        raise ValueError("Expected matched arrays of shape (N, 3), N >= 3")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Correspondence coordinates must be finite")
    return p, q


def _radius(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("inlier_radius must be finite and positive")
    return value


def _prepare_surface(points, radius):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite surface points of shape (N, 3)")
    scale = .04 / radius
    scaled = points * scale
    _, indices = np.unique(np.floor(scaled / .02).astype(np.int64),
                           axis=0, return_index=True)
    prepared = scaled[np.sort(indices)] / scale
    if len(prepared) < 6:
        raise ValueError("A surface needs at least six retained observations")
    return prepared


def _candidates(p, q, radius):
    scale, factor = .04 / radius, .1 / radius
    front = bank(p * scale, q * scale)
    R, t = front["rotation"], front["translation"] / scale
    native_index = int(front["scores"].argmax())
    native = dict(valid=True, R=R[native_index].copy(),
                  t=np.asarray(t[native_index]).copy() * factor)
    seeds = front["seeds"]
    if seeds.shape[1] == 3:
        aux_R, internal = seed_poses(p * scale, q * scale, seeds)
        aux_t = internal / scale
    else:
        aux_R, aux_t = np.empty((0, 3, 3)), np.empty((0, 3))
    return dict(R=R, t=t * factor, native=native,
                aux_R=aux_R, aux_t=aux_t * factor)


def register_correspondences(p, q, *, inlier_radius=.04):
    """Estimate q = p @ R.T + t using only tentative correspondences.

    All coordinates and inlier_radius use the same physical unit. The default
    retains the native model radius. The returned translation uses input units.
    """
    p, q = _matches(p, q)
    radius = _radius(inlier_radius)
    scale = .04 / radius
    candidates = bank(p * scale, q * scale)
    index = int(candidates["scores"].argmax())
    return {"R": candidates["rotation"][index].copy(),
            "t": candidates["translation"][index].copy() / scale}


def register_with_surfaces(p, q, source_surface, target_surface, *,
                           inlier_radius=.1, budget=16):
    """Estimate a pose using correspondences and observed source/target surfaces.

    Surfaces are finite (L, 3) observation arrays in the same unit as p and q.
    Each is reduced to its first observation per voxel of side .5*inlier_radius.
    Use radius .1 for metre-scale indoor data and .6 for metre-scale outdoor
    driving data. budget caps candidate-refinement attempts at 16 or 32. Each
    attempt uses up to two ICP capture radii. The
    returned translation uses input units; no labels or reference pose are used.
    """
    p, q = _matches(p, q)
    radius = _radius(inlier_radius)
    if not isinstance(budget, (int, np.integer)) or budget not in (16, 32):
        raise ValueError("budget must be 16 or 32")
    budget = int(budget)
    factor = .1 / radius
    source, target = (_prepare_surface(points, radius) * factor
                      for points in (source_surface, target_surface))
    candidates = _candidates(p, q, radius)
    result = process(p * factor, q * factor, source, target,
                     candidates["R"], candidates["t"], candidates["native"],
                     candidates["aux_R"], candidates["aux_t"], maximum=budget)
    pose = result["outcomes"][str(budget)]
    if not pose["valid"]:
        raise ValueError("No valid retained surface alignment")
    return {"R": np.asarray(pose["R"]).copy(),
            "t": np.asarray(pose["t"]).copy() / factor}
