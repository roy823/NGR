"""Rigid registration and reusable candidate verification in physical units."""
import numpy as np

from ._core import bank
from ._rigid import seed_poses
from ._surface import legal, process


def _matches(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or p.shape != q.shape or len(p) < 3:
        raise ValueError("Expected matched arrays of shape (N, 3), N >= 3")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Correspondence coordinates must be finite")
    return p, q


def _radius(value):
    if np.ndim(value) != 0:
        raise ValueError("inlier_radius must be a scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("inlier_radius must be finite and positive")
    return value


def _budget(value):
    if not isinstance(value, (int, np.integer)) or value not in (16, 32):
        raise ValueError("budget must be 16 or 32")
    return int(value)


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


def _poses(rotations, translations):
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or translations.shape != (len(rotations), 3)):
        raise ValueError("Expected rotations (K, 3, 3) and translations (K, 3)")
    return rotations, translations


def _candidates(p, q, radius):
    scale = .04 / radius
    front = bank(p * scale, q * scale)
    R, t = front["rotation"], front["translation"] / scale
    i = int(front["scores"].argmax())
    native = dict(R=R[i].copy(), t=t[i].copy())
    seeds = front["seeds"]
    if seeds.shape[1] == 3:
        aux_R, internal = seed_poses(p * scale, q * scale, seeds)
        aux_t = internal / scale
    else:
        aux_R, aux_t = np.empty((0, 3, 3)), np.empty((0, 3))
    return dict(R=R, t=t, native=native, aux_R=aux_R, aux_t=aux_t)


def register_correspondences(p, q, *, inlier_radius=.04):
    """Estimate q = p @ R.T + t with capacity-aware motif generation.

    Correspondences and inlier_radius use the same physical unit. Return R
    with shape (3, 3) and t with shape (3,), mapping source into target.
    """
    p, q = _matches(p, q)
    radius = _radius(inlier_radius)
    scale = .04 / radius
    candidates = bank(p * scale, q * scale)
    i = int(candidates["scores"].argmax())
    return dict(R=candidates["rotation"][i].copy(),
                t=candidates["translation"][i].copy() / scale)


def _finish(p, q, source, target, candidates, radius, budget):
    factor = .1 / radius
    native = candidates["native"]
    result = process(
        p * factor, q * factor, source * factor, target * factor,
        candidates["R"], candidates["t"] * factor,
        dict(valid=True, R=native["R"], t=native["t"] * factor),
        candidates["aux_R"], candidates["aux_t"] * factor, maximum=budget)
    i = result["index"]
    if i is None:
        raise ValueError("No valid surface alignment")
    return dict(R=result["bank"]["refined_R"][i].copy(),
                t=result["bank"]["refined_t"][i].copy() / factor,
                refinement_count=result["refinement_count"])


def refine_candidates(p, q, source_surface, target_surface,
                      rotations, translations, native_pose, *,
                      auxiliary=None, inlier_radius=.1, budget=32):
    """Allocate surface refinements and select a pose from an external pool.

    rotations and translations have shapes (K, 3, 3) and (K, 3).
    native_pose contains a proper rotation R and translation t and is retained
    in the primary shortlist. auxiliary optionally supplies another (R, t)
    pool for the extra refinement queue. All translations and point arrays
    use the input physical unit. Invalid raw candidate poses are discarded;
    a malformed pool or invalid native pose raises ValueError.

    Return R, t and refinement_count (candidate attempts, each with up to
    two ICP scales). No learned parameters or inlier labels are required.
    """
    p, q = _matches(p, q)
    radius, budget = _radius(inlier_radius), _budget(budget)
    rotations, translations = _poses(rotations, translations)
    try:
        native_R = np.asarray(native_pose["R"], dtype=np.float64)
        native_t = np.asarray(native_pose["t"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("native_pose must contain R and t arrays") from error
    if (native_R.shape != (3, 3) or native_t.shape != (3,)
            or not legal(native_R[None], native_t[None])[0]):
        raise ValueError("native_pose must be a finite proper rigid transformation")
    if auxiliary is None:
        aux_R, aux_t = np.empty((0, 3, 3)), np.empty((0, 3))
    else:
        aux_R, aux_t = _poses(*auxiliary)
    source, target = (_prepare_surface(points, radius)
                      for points in (source_surface, target_surface))
    candidates = dict(R=rotations, t=translations, native=dict(R=native_R, t=native_t),
                      aux_R=aux_R, aux_t=aux_t)
    return _finish(p, q, source, target, candidates, radius, budget)


def register_with_surfaces(p, q, source_surface, target_surface, *,
                           inlier_radius=.1, budget=32):
    """Run motif generation, retained growth, surface finishing and release.

    Surfaces and correspondences use the same physical unit. At most budget
    candidates are refined. Each surface retains its first observation in
    each voxel of side .5*inlier_radius, preserving observation order.
    """
    p, q = _matches(p, q)
    radius, budget = _radius(inlier_radius), _budget(budget)
    source, target = (_prepare_surface(points, radius)
                      for points in (source_surface, target_surface))
    front = _candidates(p, q, radius)
    return _finish(p, q, source, target, front, radius, budget)
