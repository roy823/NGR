"""Observed-surface evidence, candidate selection and point-to-plane ICP."""
from time import perf_counter

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


def empty_pose():
    return dict(valid=False,R=None,t=None)


def pose_at(post, index):
    if index is None or not post['valid'][index]:
        return empty_pose()
    return dict(valid=True,R=post['refined_R'][index].tolist(),t=post['refined_t'][index].tolist())


def choose(post, information, keep, budget):
    stop=min(budget,len(post['valid']));retained=np.flatnonzero(post['valid'][:min(keep,stop)])
    if not len(retained):return None
    M,G=post['refined_M'],post['refined_G'];anchor=int(retained[np.argmax(M[retained]*np.sqrt(G[retained]))])
    score=M[:stop]*np.sqrt(information[:stop])
    permitted=np.flatnonzero(post['valid'][:stop]&(information[:stop]>=information[anchor])&(score>score[anchor]))
    return int(permitted[np.argmax(score[permitted])]) if len(permitted) else anchor


def process(p,q,source,target,raw_R,raw_t,native,aux_R=None,aux_t=None,maximum=32):
    if maximum not in (16,32):raise ValueError('Budgets must be 16 or 32')
    began=perf_counter();timing={};reuse=0;fresh=0;stages=[]
    R,t=np.asarray(raw_R,dtype=np.float64),np.asarray(raw_t,dtype=np.float64)
    if not native.get('valid',False):return dict(outcomes={'ICP1':empty_pose(),'16':empty_pose(),'32':empty_pose()},status='no_native_pose',seconds=perf_counter()-began)
    if aux_R is None:aux_R,aux_t=np.empty((0,3,3)),np.empty((0,3))
    initial=shortlist(p,q,R,t,np.asarray(native['R']),np.asarray(native['t']))
    evidence=Evidence(source,target);backend=None
    def polish(r,translation):
        nonlocal backend,reuse,fresh
        M=mae(p,q,r,translation);G=evidence.score(r,translation);parents=allocation(M,G)
        if backend is None:backend=SurfaceICP(source,target)
        fit_R=np.zeros((len(parents),3,3));fit_t=np.zeros((len(parents),3));valid=np.zeros(len(parents),dtype=bool)
        for index,parent in enumerate(parents):
            prediction,trace=backend.apply(dict(valid=True,R=r[parent],t=translation[parent]));fresh+=1;stages.append(trace)
            if prediction['valid']:fit_R[index],fit_t[index],valid[index]=prediction['R'],prediction['t'],True
        post_M,post_G=np.zeros(len(parents)),np.zeros(len(parents))
        if valid.any():post_M[valid]=mae(p,q,fit_R[valid],fit_t[valid]);post_G[valid]=evidence.score(fit_R[valid],fit_t[valid])
        return dict(R=r,t=translation,parents=parents,refined_R=fit_R,refined_t=fit_t,refined_M=post_M,refined_G=post_G,valid=valid)
    base=polish(initial['R'],initial['t']);keep=len(base['parents']);timing['base_seconds']=perf_counter()-began;mark=perf_counter()
    raw_ids=np.empty(0,dtype=int);extra_order=np.empty(0,dtype=int);screened=0;skyline_count=0
    if maximum>keep:
        proposals_R=np.concatenate((R,aux_R));proposals_t=np.concatenate((t,aux_t))
        valid=legal(proposals_R,proposals_t);raw_ids=np.flatnonzero(valid);proposals_R,proposals_t=proposals_R[valid],proposals_t[valid]
        M=mae(p,q,proposals_R,proposals_t);I=evidence.score(proposals_R,proposals_t,information=True,early=True);screened=len(M)
        parents=base['parents'];unused=distinct(proposals_R,proposals_t,np.argsort(-I,kind='stable'),base['R'][parents],base['t'][parents],maximum=len(proposals_R))
        skyline=unused[frontier(M[unused],I[unused])];skyline_count=len(skyline)
        selected=skyline[np.argsort(-I[skyline],kind='stable')[:16]]
        # The complete extra queue is ranked before cutting the global budget.
        # This makes B16 and B32 genuine nested caps even when keep<16.
        if len(selected):
            extra_R,extra_t=proposals_R[selected],proposals_t[selected]
            extra_M=mae(p,q,extra_R,extra_t);extra_G=evidence.score(extra_R,extra_t);order=allocation(extra_M,extra_G)
            extra_order=order[:max(0,maximum-keep)];extra_R,extra_t=extra_R[extra_order],extra_t[extra_order]
            extra_order=raw_ids[selected[extra_order]]
        else:extra_R,extra_t=np.empty((0,3,3)),np.empty((0,3))
    else:extra_R,extra_t=np.empty((0,3,3)),np.empty((0,3))
    timing['screen_seconds']=perf_counter()-mark;mark=perf_counter()
    if len(extra_R):
        # The queue already carries allocation order. Preserve it while fitting.
        if backend is None:backend=SurfaceICP(source,target)
        fit_R=np.zeros_like(extra_R);fit_t=np.zeros_like(extra_t);valid=np.zeros(len(extra_R),dtype=bool)
        for index in range(len(extra_R)):
            pred,trace=backend.apply(dict(valid=True,R=extra_R[index],t=extra_t[index]));fresh+=1;stages.append(trace)
            if pred['valid']:fit_R[index],fit_t[index],valid[index]=pred['R'],pred['t'],True
        M,G=np.zeros(len(extra_R)),np.zeros(len(extra_R))
        if valid.any():M[valid]=mae(p,q,fit_R[valid],fit_t[valid]);G[valid]=evidence.score(fit_R[valid],fit_t[valid])
        extra=dict(refined_R=fit_R,refined_t=fit_t,refined_M=M,refined_G=G,valid=valid)
        post={key:np.concatenate((base[key],extra[key])) for key in ('refined_R','refined_t','refined_M','refined_G','valid')}
    else:post={key:base[key].copy() for key in ('refined_R','refined_t','refined_M','refined_G','valid')}
    timing['extra_ICP_seconds']=perf_counter()-mark;mark=perf_counter()
    information=evidence.score(post['refined_R'],post['refined_t'],information=True)
    outcomes={'ICP1':pose_at(base,0)};indices={}
    for budget in (16,32):
        if budget>maximum:continue
        indices[str(budget)]=choose(post,information,keep,budget);outcomes[str(budget)]=pose_at(post,indices[str(budget)])
    timing['readout_seconds']=perf_counter()-mark
    post.update(information_G=information,retained_ICP_count=np.array(keep),extra_initial_R=extra_R,extra_initial_t=extra_t,selected_raw_indices=extra_order)
    return dict(outcomes=outcomes,indices=indices,status='complete',bank=post,screened=screened,skyline_modes=skyline_count,retained_calls=keep,total_calls=len(post['valid']),
        fresh_ICP_calls=fresh,reused_ICP_calls=reuse,ICP_stages=stages,timing=timing,seconds=perf_counter()-began)
