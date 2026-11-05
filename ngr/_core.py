"""Geometric compatibility, motif initialization and phase growth."""
import heapq
import numpy as np
import torch
from scipy.linalg.blas import ssyrk
from scipy.spatial.distance import pdist,squareform
from . import _rigid as reference


def capacity_greedy(items, affinity, count, capacity, occupied):
    """Select motifs by affinity times the vacancies of their occupied sites."""
    used = {}
    excluded = set(occupied)
    for i in excluded:
        for x in items[i]:
            used[int(x)] = used.get(int(x), 0) + 1
    heap = [(-float(value), i) for i, value in enumerate(affinity)
            if i not in excluded]
    heapq.heapify(heap)
    selected = []
    while heap and len(selected) < count:
        negative, i = heapq.heappop(heap)
        value = float(affinity[i]) * np.prod(
            [1 / (1 + used.get(int(x), 0) / capacity) for x in items[i]])
        if value < -negative - 1e-15:
            heapq.heappush(heap, (-value, i))
            continue
        selected.append(i)
        for x in items[i]:
            used[int(x)] = used.get(int(x), 0) + 1
    return selected


def capacity_pairs(graph, square, count):
    available = len(graph) * (len(graph) - 1) // 2
    count = min(count, available)
    wide = pair_shortlist(graph, square, min(65536, available))
    a, b = wide.numpy()
    affinity = (graph[wide[0], wide[1]] * square[wide[0], wide[1]]).numpy()
    head = list(range(count // 2))
    tail = capacity_greedy(np.stack((a, b), 1), affinity,
                           count - len(head), 8, head)
    return wide[:, torch.as_tensor(head + tail, dtype=torch.long)]

def pair_shortlist(graph, square, count=4096):
    a, b = torch.triu_indices(len(graph), len(graph), 1)
    support = square[a, b]
    score = graph[a, b] * support
    count = min(count, len(score))
    threshold = torch.topk(score, count, sorted=False).values.min()
    above = torch.nonzero(score > threshold).flatten()
    equal = torch.nonzero(score == threshold).flatten()
    selected = torch.cat((above, equal[:count - len(above)]))
    selected = selected[torch.argsort(score[selected], descending=True, stable=True)]
    return torch.stack((a[selected], b[selected]))

def triplet_weights(graph, triples):
    a, b, c = torch.as_tensor(triples, dtype=torch.long).T
    weights = graph[a] * graph[b] * graph[c]
    rows = torch.arange(len(a))
    weights[rows, a] += 1
    weights[rows, b] += 1
    weights[rows, c] += 1
    return weights / weights.sum(-1, keepdim=True)

def nuclei(graph, a, b):
    left = torch.as_tensor(a, dtype=torch.long, device=graph.device)
    right = torch.as_tensor(b, dtype=torch.long, device=graph.device)
    weights = graph[left] * graph[right]
    rows = torch.arange(len(left), device=graph.device)
    weights[rows, left] += 1.0
    weights[rows, right] += 1.0
    return weights / weights.sum(-1, keepdim=True)
def grow(weights, graph, stages=10):
    n = len(graph)
    for _ in range(stages):
        support = 1 + n * (weights @ graph)
        weights = (weights + 1e-30) * support.square()
        weights = weights / weights.sum(-1, keepdim=True)
    return weights
def distance_graph(p, q):
    left, right = (pdist(p.numpy()), pdist(q.numpy()))
    delta = torch.from_numpy(right - left)
    values = torch.exp(-0.5 * (delta / 0.015).square())
    graph = torch.from_numpy(squareform(values.numpy()))
    n = len(p)
    load = float((n - 2) * (graph.sum() / (n * (n - 1))) ** 2)
    graph = graph.to(torch.float32)
    if load < 1:
        return (graph, None, None, load)
    return (graph, torch.from_numpy(squareform(left * left)), torch.from_numpy(squareform(right * right)), load)

def shape_mask(edges, a, b):
    u, v, w = (edges[a, b, None], edges[a], edges[b])
    total = u + v + w
    dot = (u + v - w) * 0.5
    return (3 * (u * v - dot.square()) >= 0.01 / 1.01 ** 2 * total.square()) & (total > 0)

def triplets(graph, source_edges, target_edges, pairs):
    buffers = []
    n = len(graph)
    for offset in range(0, pairs.shape[1], 256):
        a, b = pairs[:, offset:offset + 256]
        common = graph[a] * graph[b]
        rows = torch.arange(len(a))
        common[rows, a] = 0
        common[rows, b] = 0
        score = common * (common @ graph) * graph[a, b, None]
        shape = shape_mask(source_edges, a, b) & shape_mask(target_edges, a, b)
        score.masked_fill_(~shape, -torch.inf)
        values, flat = torch.topk(score.flatten(), min(3072, score.numel()), sorted=False)
        positive = values > 0
        values, flat = (values[positive], flat[positive])
        row, c = (flat // n, flat % n)
        buffers.append((torch.stack((a[row], b[row], c), -1).numpy(), values.numpy()))
    triples = np.concatenate([b[0] for b in buffers])
    values = np.concatenate([b[1] for b in buffers])
    _, first = np.unique(np.sort(triples, axis=1), axis=0, return_index=True)
    order = first[np.argsort(-values[first], kind='stable')[:16384]]
    deep, affinity = triples[order], values[order]
    head = list(range(min(512, len(deep))))
    tail = capacity_greedy(deep, affinity, 1024 - len(head), 4, head)
    return deep[np.asarray(head + tail, dtype=np.int64)]

@torch.inference_mode()
def bank(p,q):
    p,q=(torch.as_tensor(x,dtype=torch.float64) for x in (p,q))
    graph,left,right,load=distance_graph(p,q)
    upper=torch.from_numpy(ssyrk(1.,graph.numpy().T,trans=1,lower=0))
    pairs=capacity_pairs(graph,upper,256 if load<1 else 4096)
    del upper
    if load<1:
        seeds=pairs.T.numpy();initial=nuclei(graph,*pairs);mode='count'
    else:
        seeds=triplets(graph,left,right,pairs)
        if not len(seeds):raise ValueError('No nondegenerate nucleus')
        initial=triplet_weights(graph,seeds);mode='mae'
    del left,right
    grown=grow(initial,graph)
    del graph,initial
    rotation,translation=reference.fit_and_refine(p,q,grown.to(torch.float64))
    residual=reference.residuals(rotation,translation,p,q)
    scores=(residual<.04).sum(-1)-.001*residual.mean(-1) if mode=='count' else torch.clamp(1-residual/.04,min=0).sum(-1)
    return dict(rotation=rotation.numpy(),translation=translation.numpy(),scores=scores.numpy(),seeds=seeds,score_name=mode)
