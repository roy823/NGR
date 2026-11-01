"""Observed-surface evidence, candidate selection and point-to-plane ICP."""
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

def geometry_score(source, target, source_query, target_query, rotation, translation, radius):
    target_tree, source_tree = (cKDTree(target), cKDTree(source))
    forward = target_tree.query(source_query[None] @ rotation.transpose(0, 2, 1) + translation[:, None], workers=1)[0]
    reverse = source_tree.query((target_query[None] - translation[:, None]) @ rotation, workers=1)[0]
    a = np.maximum(1 - forward / radius, 0).mean(-1)
    b = np.maximum(1 - reverse / radius, 0).mean(-1)
    return (2 * a * b / (a + b + 1e-12), a, b)
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

def harmonic(a, b):
    return 2 * a * b / (a + b + 1e-12)

def geometry(source, target, sq, tq, R, t, sw, tw):
    forward = cKDTree(target).query(sq[None] @ R.transpose(0, 2, 1) + t[:, None], workers=1)[0]
    reverse = cKDTree(source).query((tq[None] - t[:, None]) @ R, workers=1)[0]
    a = np.maximum(1 - forward / 0.1, 0)
    b = np.maximum(1 - reverse / 0.1, 0)
    return (harmonic(a @ sw, b @ tw), harmonic(a.mean(-1), b.mean(-1)))
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
