"""Public coordinate-to-pose interfaces."""
import numpy as np

from ._core import bank
from ._rigid import seed_poses
from ._surface import (
    SurfaceICP, allocation, distinct, frontier, geometry, geometry_score,
    mae, query_weights, sampled_points, shortlist,
)


def _matches(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.ndim != 2 or p.shape[1:] != (3,) or p.shape != q.shape or len(p) < 3:
        raise ValueError("Expected matched arrays of shape (N, 3), N >= 3")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Correspondence coordinates must be finite")
    return p, q


def register_correspondences(p, q):
    """Estimate q = p @ R.T + t in native model units (radius 0.04)."""
    p, q = _matches(p, q)
    candidates = bank(p, q)
    index = int(candidates["scores"].argmax())
    return {"R": candidates["rotation"][index].copy(),
            "t": candidates["translation"][index].copy()}


def prepare_indoor_surface(points):
    """Keep the first observed point in each 0.05-metre voxel, in input order."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Expected finite surface points of shape (N, 3)")
    _, indices = np.unique(np.floor((points * (.04 / .1)) / .02).astype(np.int64),
                           axis=0, return_index=True)
    return points[np.sort(indices)].copy()


def _polish(p, q, source, target, backend, rotation, translation):
    M = mae(p, q, rotation, translation)
    _, sq = sampled_points(source)
    _, tq = sampled_points(target)
    G = geometry_score(source, target, sq, tq, rotation, translation, .1)[0]
    parents = allocation(M, G)
    R = np.zeros((len(parents), 3, 3))
    t = np.zeros((len(parents), 3))
    valid = np.zeros(len(parents), dtype=bool)
    for position, parent in enumerate(parents):
        pose, _ = backend.apply(dict(valid=True, R=rotation[parent], t=translation[parent]))
        if pose["valid"]:
            R[position], t[position], valid[position] = pose["R"], pose["t"], True
    post_M, post_G = np.zeros(len(parents)), np.zeros(len(parents))
    if valid.any():
        post_M[valid] = mae(p, q, R[valid], t[valid])
        post_G[valid] = geometry_score(source, target, sq, tq, R[valid], t[valid], .1)[0]
    return dict(R=rotation, t=translation, parents=parents, refined_R=R, refined_t=t,
                refined_M=post_M, refined_G=post_G, valid=valid)


def register_indoor(p, q, source_surface, target_surface):
    """Register indoor matches and prepared observed surfaces, all in metres."""
    p, q = _matches(p, q)
    source, target = (np.asarray(x, dtype=np.float64)
                      for x in (source_surface, target_surface))
    scale = .04 / .1
    candidates = bank(p * scale, q * scale)
    if candidates["score_name"] != "mae":
        raise ValueError("Indoor surface verification requires three-point nuclei")
    R, t = candidates["rotation"], candidates["translation"] / scale
    native = int(candidates["scores"].argmax())
    initial = shortlist(p, q, R, t, R[native], t[native])
    backend = SurfaceICP(source, target)
    base = _polish(p, q, source, target, backend, initial["R"], initial["t"])
    _, sq = sampled_points(source)
    _, tq = sampled_points(target)
    sw, tw = query_weights(sq), query_weights(tq)
    anchor_R, anchor_t = seed_poses(p * scale, q * scale, candidates["seeds"])
    proposals_R = np.concatenate((R, anchor_R))
    proposals_t = np.concatenate((t, anchor_t / scale))
    M = mae(p, q, proposals_R, proposals_t)
    I, _ = geometry(source, target, sq[:128], tq[:128], proposals_R, proposals_t,
                    sw[:128] / sw[:128].sum(), tw[:128] / tw[:128].sum())
    visited = base["parents"]
    unused = distinct(proposals_R, proposals_t, np.argsort(-I, kind="stable"),
                      base["R"][visited], base["t"][visited], maximum=len(proposals_R))
    skyline = unused[frontier(M[unused], I[unused])]
    selected = skyline[np.argsort(-I[skyline], kind="stable")[:16]]
    keep = len(base["refined_R"])
    if len(selected):
        extra = _polish(p, q, source, target, backend, proposals_R[selected], proposals_t[selected])
        post = {key: np.concatenate((base[key], extra[key])) for key in
                ("refined_R", "refined_t", "refined_M", "refined_G", "valid")}
    else:
        post = {key: base[key] for key in
                ("refined_R", "refined_t", "refined_M", "refined_G", "valid")}
    information, _ = geometry(source, target, sq, tq, post["refined_R"],
                              post["refined_t"], sw, tw)
    retained = np.flatnonzero(post["valid"][:keep])
    if not len(retained):
        raise ValueError("No valid retained surface alignment")
    incumbent = int(retained[np.argmax(post["refined_M"][retained] *
                                      np.sqrt(post["refined_G"][retained]))])
    score = post["refined_M"] * np.sqrt(information)
    eligible = np.flatnonzero(post["valid"] & (information >= information[incumbent]) &
                              (score > score[incumbent]))
    winner = int(eligible[np.argmax(score[eligible])]) if len(eligible) else incumbent
    return {"R": post["refined_R"][winner].copy(), "t": post["refined_t"][winner].copy()}
