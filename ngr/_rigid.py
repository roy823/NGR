"""Batched rigid fits and correspondence refitting."""
import numpy as np
import torch

def kabsch(p, q, weights):
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-12)
    pc, qc = (weights @ p, weights @ q)
    P, Q = (p[None] - pc[:, None], q[None] - qc[:, None])
    covariance = (P * weights[:, :, None]).transpose(-1, -2) @ Q
    covariance = covariance + 1e-09 * torch.eye(3, dtype=p.dtype, device=p.device)
    U, _, Vh = torch.linalg.svd(covariance, full_matrices=False)
    V, Ut = (Vh.transpose(-1, -2), U.transpose(-1, -2))
    sign = torch.sign(torch.linalg.det(V @ Ut))
    diagonal = torch.stack((torch.ones_like(sign), torch.ones_like(sign), sign), dim=-1)
    rotation = V * diagonal[:, None, :] @ Ut
    translation = qc - (rotation @ pc[:, :, None]).squeeze(-1)
    return (rotation, translation)

def residuals(rotation, translation, p, q):
    return torch.linalg.vector_norm(p[None] @ rotation.transpose(-1, -2) + translation[:, None] - q[None], dim=-1)

def fit_and_refine(p, q, weights):
    rotation, translation = kabsch(p, q, weights)
    for _ in range(2):
        residual = residuals(rotation, translation, p, q)
        weights = (residual < 0.04).to(p.dtype) + 1e-09
        rotation, translation = kabsch(p, q, weights)
    return (rotation, translation)
@torch.inference_mode()
def seed_poses(p, q, seeds):
    p, q = (torch.as_tensor(x, dtype=torch.float64) for x in (p, q))
    seeds = torch.as_tensor(seeds, dtype=torch.long)
    if seeds.ndim != 2 or seeds.shape[1] != 3:
        raise ValueError('Nondegenerate three-point nuclei required')
    weights = torch.ones((1, 3), dtype=torch.float64)
    R, t = torch.vmap(lambda a, b: kabsch(a, b, weights))(p[seeds], q[seeds])
    return (R[:, 0].numpy(), t[:, 0].numpy())
