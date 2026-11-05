"""Budgeted surface refinement, geometric evidence and complementary release."""

import numpy as np
from scipy.spatial import cKDTree

def correspondence_score(p, q, rotation, translation):
    residual = np.linalg.norm(p[None] @ rotation.transpose(0, 2, 1) + translation[:, None] - q[None], axis=-1)
    return np.maximum(1 - residual / .1, 0).sum(-1)

def sampled_points(points, maximum=4096):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6 or (not np.isfinite(points).all()):
        raise ValueError('Expected at least six finite surface points')
    order = np.random.default_rng(3107).permutation(len(points))
    split = len(points) // 2
    return (points[order[:min(maximum, split)]], points[order[split:split + maximum]])


def normals(points):
    if len(points) < 3:
        raise ValueError('A surface patch needs at least three points')
    _, indices = cKDTree(points).query(points, k=min(20, len(points)), workers=1)
    local = points[indices]
    local = local - local.mean(1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', local, local)
    _, vectors = np.linalg.eigh(covariance)
    return vectors[:, :, 0]
def information_weights(points, normal):
    points, normal = (np.asarray(points, dtype=np.float64), np.asarray(normal, dtype=np.float64))
    centered = points - points.mean(0)
    scale = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if scale <= 0 or points.shape != normal.shape:
        raise ValueError('Nonzero finite surface extent required')
    J = np.c_[np.cross(centered / scale, normal), normal]
    inverse = np.linalg.pinv(J.T @ J, hermitian=True)
    leverage = np.maximum(np.einsum('ni,ni->n', J @ inverse, J), 0)
    if not np.isfinite(leverage).all() or leverage.sum() <= 0:
        raise ValueError('Invalid geometric information')
    return leverage / leverage.sum()

def query_weights(query):
    return information_weights(query, normals(query))




def legal(rotation, translation):
    R, t = (np.asarray(rotation), np.asarray(translation))
    if R.ndim != 3 or R.shape[1:] != (3, 3) or t.shape != (len(R), 3):
        raise ValueError('Malformed pose bank')
    finite = np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1)
    result = np.zeros(len(R), dtype=bool)
    valid = np.flatnonzero(finite)
    if len(valid):
        result[valid] = (np.max(np.abs(R[valid].transpose(0, 2, 1) @ R[valid] - np.eye(3)), axis=(1, 2)) <= 1e-05) & (np.linalg.det(R[valid]) > 0)
    return result

def mae(p, q, rotation, translation):
    output = []
    for start in range(0, len(rotation), 64):
        output.append(correspondence_score(p, q, rotation[start:start + 64], translation[start:start + 64]))
    return np.concatenate(output) if output else np.empty(0)

def shortlist(p, q, raw_R, raw_t, native_R, native_t):
    raw_R, raw_t = (np.asarray(raw_R, dtype=np.float64), np.asarray(raw_t, dtype=np.float64))
    valid = legal(raw_R, raw_t)
    if not legal(np.asarray(native_R)[None], np.asarray(native_t)[None])[0]:
        raise ValueError('The native final winner is not a rigid pose')
    indices = np.flatnonzero(valid)
    R = np.concatenate((np.asarray(native_R)[None], raw_R[valid]))
    t = np.concatenate((np.asarray(native_t)[None], raw_t[valid]))
    M = mae(p, q, R, t)
    selected = [0]
    for index in 1 + np.argsort(-M[1:], kind='stable'):
        cosine = (np.sum(R[selected] * R[index], axis=(1, 2)) - 1) / 2
        near = (cosine > np.cos(np.deg2rad(5))) & (np.linalg.norm(t[selected] - t[index], axis=1) < 0.1)
        if not near.any():
            selected.append(int(index))
        if len(selected) == 32:
            break
    raw_indices = np.r_[-1, indices][selected]
    return dict(R=R[selected], t=t[selected], M=M[selected], raw_indices=raw_indices, raw_count=len(raw_R), legal_raw_count=int(valid.sum()), raw_M=M[1:])
def allocation(M, G):
    if len(M) == 0 or len(M) != len(G):
        raise ValueError('Empty or inconsistent shared shortlist')
    selected = [0]
    rankings = [np.argsort(-score, kind='stable') for score in (M, G, M * G)]
    for line in zip(*rankings):
        for index in line:
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == min(16, len(M)):
                return np.asarray(selected)
    return np.asarray(selected)
def frontier(M, G):
    M, G = (np.asarray(M), np.asarray(G))
    if M.shape != G.shape or M.ndim != 1 or (not np.isfinite(M).all()) or (not np.isfinite(G).all()):
        raise ValueError('Expected finite matched one-dimensional scores')
    order = np.lexsort((np.arange(len(M)), -G, -M))
    best = -np.inf
    selected = []
    for i in order:
        if G[i] > best:
            selected.append(int(i))
            best = G[i]
    return np.asarray(selected, dtype=int)

def distinct(R, t, order, retained_R, retained_t, maximum=16):
    selected = []
    count = len(retained_R)
    capacity = count + min(maximum, len(order))
    reference_R = np.empty((capacity, 3, 3), dtype=R.dtype)
    reference_t = np.empty((capacity, 3), dtype=t.dtype)
    reference_R[:count] = retained_R
    reference_t[:count] = retained_t
    cosine = np.cos(np.deg2rad(5))
    for i in order:
        if count:
            close = ((np.sum(reference_R[:count] * R[i], axis=(1, 2)) - 1) / 2 > cosine) & (np.linalg.norm(reference_t[:count] - t[i], axis=1) < 0.1)
            if close.any():
                continue
        selected.append(int(i))
        reference_R[count] = R[i]
        reference_t[count] = t[i]
        count += 1
        if len(selected) == maximum:
            break
    return np.asarray(selected, dtype=int)
class SurfaceICP:

    def __init__(self, source, target):
        import open3d as o3d
        self.registration = o3d.pipelines.registration
        self.source = o3d.geometry.PointCloud()
        self.target = o3d.geometry.PointCloud()
        fit, _ = sampled_points(source)
        self.source.points = o3d.utility.Vector3dVector(fit)
        self.target.points = o3d.utility.Vector3dVector(target.copy())
        self.target.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=min(20, len(target))))
        self.estimator = self.registration.TransformationEstimationPointToPlane(self.registration.TukeyLoss(k=0.1))
        self.criteria = self.registration.ICPConvergenceCriteria(max_iteration=10)
        self.fit_points = len(fit)

    def apply(self, initial):
        if not initial['valid']:
            return (dict(valid=False, reason='InvalidNativePose'), [])
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = (initial['R'], initial['t'])
        stages = []
        for threshold in (0.2, 0.1):
            result = self.registration.registration_icp(self.source, self.target, threshold, matrix, self.estimator, self.criteria)
            matrix = result.transformation
            stages.append(dict(radius_m=threshold, fitness=float(result.fitness), inlier_rmse=float(result.inlier_rmse), matches=len(result.correspondence_set)))
            if not np.isfinite(matrix).all():
                return (dict(valid=False, reason='NonfiniteICP'), stages)
        rotation = matrix[:3, :3]
        valid = np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-05) and np.linalg.det(rotation) > 0
        return (dict(valid=bool(valid), R=rotation.tolist(), t=matrix[:3, 3].tolist()), stages)


class Evidence:
    def __init__(self, source, target):
        self.source, self.target = source, target
        _, self.sq = sampled_points(source)
        _, self.tq = sampled_points(target)
        self.source_tree, self.target_tree = cKDTree(source), cKDTree(target)
        self.sw = self.tw = None

    def weights(self):
        if self.sw is None:
            self.sw, self.tw = query_weights(self.sq), query_weights(self.tq)

    def moments(self, rotations, translations):
        """Counts and squared distances below the radius in both directions."""
        hits = np.zeros((len(rotations), 2), dtype=np.int64)
        squared = np.zeros((len(rotations), 2))
        for i, (rotation, translation) in enumerate(zip(rotations, translations)):
            forward = self.target_tree.query(
                self.sq @ rotation.T + translation, distance_upper_bound=.1,
                workers=1)[0]
            reverse = self.source_tree.query(
                (self.tq - translation) @ rotation, distance_upper_bound=.1,
                workers=1)[0]
            for j, distance in enumerate((forward, reverse)):
                inside = distance[distance < .1]
                hits[i, j] = len(inside)
                squared[i, j] = float(np.sum(inside**2))
        return hits, squared

    def score(self, R, t, information=False, early=False):
        if information:
            self.weights()
        sq, tq = (self.sq[:128], self.tq[:128]) if early else (self.sq, self.tq)
        if information:
            sw, tw = (self.sw[:128]/self.sw[:128].sum(), self.tw[:128]/self.tw[:128].sum()) if early else (self.sw, self.tw)
        scores = []
        # Bound memory without truncating any pose. NGR's entire <=2048 pool
        # remains a single matrix; larger pose pools are evaluated in blocks.
        for offset in range(0, len(R), 8192):
            r, translation = R[offset:offset+8192], t[offset:offset+8192]
            forward = self.target_tree.query(sq[None]@r.transpose(0,2,1)+translation[:,None],workers=1)[0]
            reverse = self.source_tree.query((tq[None]-translation[:,None])@r,workers=1)[0]
            a, b = np.maximum(1-forward/.1,0), np.maximum(1-reverse/.1,0)
            x, y = (a@sw, b@tw) if information else (a.mean(-1), b.mean(-1))
            scores.append(2*x*y/(x+y+1e-12))
        return np.concatenate(scores) if scores else np.empty(0)


def concentration_score(mean_squared_radius):
    """One-sided concentration rate for normalized squared surface distances."""
    q = np.asarray(mean_squared_radius, dtype=np.float64)
    score = np.zeros_like(q)
    score[q <= 0] = np.inf
    inside = (q > 0) & (q < .5)
    if inside.any():
        target = q[inside]
        lo, hi = np.zeros_like(target), 1 / target
        for _ in range(100):
            mid = (lo + hi) / 2
            small = mid < 1e-3
            with np.errstate(over="ignore"):
                mean = np.where(small, .5 - mid/12 + mid**3/720,
                                1/np.where(small, 1., mid)
                                - 1/np.expm1(np.where(small, 1., mid)))
            larger = mean > target
            lo = np.where(larger, mid, lo)
            hi = np.where(larger, hi, mid)
        tilt = (lo + hi) / 2
        score[inside] = -tilt*target - (np.log(-np.expm1(-tilt)) - np.log(tilt))
    return score


def release_index(support, hits, squared_distances):
    """Select by the weaker normalized support and residual-concentration score."""
    support = np.asarray(support, dtype=np.float64)
    hits = np.asarray(hits)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = squared_distances / (hits * .1**2)
    evidence = concentration_score(np.where(hits > 0, mean, 1.)).mean(1)
    evidence[(hits == 0).any(1)] = -np.inf
    if np.isneginf(evidence).all():
        return int(np.argmax(support))
    if np.isposinf(evidence).any():
        shape = np.isposinf(evidence).astype(float)
    else:
        shape = np.exp(evidence - evidence.max())
    if support.max() <= 0:
        return int(np.argmax(shape))
    support = support / support.max()
    score = np.minimum(support, shape)
    return int(np.lexsort((-np.arange(len(score)), shape, support, score))[-1])


def process(p, q, source, target, raw_R, raw_t, native,
            aux_R=None, aux_t=None, maximum=32):
    """Allocate a refinement queue and release a pose; coordinates use radius .1."""
    if maximum not in (16, 32):
        raise ValueError("Budget must be 16 or 32")
    if not native.get("valid", False):
        raise ValueError("A valid native pose is required")
    R, t = np.asarray(raw_R, dtype=np.float64), np.asarray(raw_t, dtype=np.float64)
    if aux_R is None:
        aux_R, aux_t = np.empty((0, 3, 3)), np.empty((0, 3))
    initial = shortlist(p, q, R, t, np.asarray(native["R"]), np.asarray(native["t"]))
    evidence = Evidence(source, target)
    backend = SurfaceICP(source, target)

    def refine(rotations, translations):
        fitted_R = np.zeros_like(rotations)
        fitted_t = np.zeros_like(translations)
        valid = np.zeros(len(rotations), dtype=bool)
        for i in range(len(rotations)):
            pose, _ = backend.apply(dict(valid=True, R=rotations[i], t=translations[i]))
            if pose["valid"]:
                fitted_R[i], fitted_t[i], valid[i] = pose["R"], pose["t"], True
        M, G = np.zeros(len(rotations)), np.zeros(len(rotations))
        if valid.any():
            M[valid] = mae(p, q, fitted_R[valid], fitted_t[valid])
            G[valid] = evidence.score(fitted_R[valid], fitted_t[valid])
        return dict(refined_R=fitted_R, refined_t=fitted_t,
                    refined_M=M, refined_G=G, valid=valid)

    M = mae(p, q, initial["R"], initial["t"])
    G = evidence.score(initial["R"], initial["t"])
    parents = allocation(M, G)
    base = refine(initial["R"][parents], initial["t"][parents])
    primary_count = len(parents)
    extra_R, extra_t = np.empty((0, 3, 3)), np.empty((0, 3))
    selected_raw_indices = np.empty(0, dtype=int)
    if maximum > primary_count:
        proposal_R = np.concatenate((R, aux_R))
        proposal_t = np.concatenate((t, aux_t))
        ids = np.flatnonzero(legal(proposal_R, proposal_t))
        proposal_R, proposal_t = proposal_R[ids], proposal_t[ids]
        M = mae(p, q, proposal_R, proposal_t)
        I = evidence.score(proposal_R, proposal_t, information=True, early=True)
        unused = distinct(proposal_R, proposal_t, np.argsort(-I, kind="stable"),
                          initial["R"][parents], initial["t"][parents],
                          maximum=len(proposal_R))
        skyline = unused[frontier(M[unused], I[unused])]
        selected = skyline[np.argsort(-I[skyline], kind="stable")[:16]]
        if len(selected):
            extra_R, extra_t = proposal_R[selected], proposal_t[selected]
            extra_M = mae(p, q, extra_R, extra_t)
            extra_G = evidence.score(extra_R, extra_t)
            queue = allocation(extra_M, extra_G)[:maximum - primary_count]
            extra_R, extra_t = extra_R[queue], extra_t[queue]
            selected_raw_indices = ids[selected[queue]]

    if len(extra_R):
        extra = refine(extra_R, extra_t)
        post = {key: np.concatenate((base[key], extra[key])) for key in base}
    else:
        post = {key: value.copy() for key, value in base.items()}
    ids = np.flatnonzero(post["valid"])
    information = np.zeros(len(post["valid"]))
    hit_counts = np.zeros((len(post["valid"]), 2), dtype=np.int64)
    squared = np.zeros((len(post["valid"]), 2))
    index = None
    if len(ids):
        information[ids] = evidence.score(post["refined_R"][ids],
                                           post["refined_t"][ids], information=True)
        hit_counts[ids], squared[ids] = evidence.moments(post["refined_R"][ids],
                                                       post["refined_t"][ids])
        support = post["refined_M"][ids] * np.sqrt(information[ids])
        index = int(ids[release_index(support, hit_counts[ids], squared[ids])])
    post.update(information_G=information, moment_counts=hit_counts,
                moment_squared=squared, extra_initial_R=extra_R, extra_initial_t=extra_t,
                selected_raw_indices=selected_raw_indices)
    return dict(index=index, bank=post, primary_count=primary_count,
                refinement_count=len(post["valid"]))
