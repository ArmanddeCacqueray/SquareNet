"""
PyTorch implementation of Cartesian sort
GPU compatible

Typical usage
-------------
>>> def step(g, pts_flat):
...     g, lc = torch_cartesian_sort(g, pts_flat)
...     pts_grid = pts_flat[g]  # (*G, D)
...
...     # apply some logic on gridified points...
...     # ... and back to flat structure
...     pts_flat = pts_grid.reshape(-1, D)  # (N, D)
...     return g, pts_flat
"""

import torch


# ============================================================================
# Helpers
# ============================================================================

def _active_dims(g):
    shape = list(g.shape)
    dims = [i for i, s in enumerate(shape) if s > 1]
    dims.sort(key=lambda d: -shape[d])
    return tuple(dims)


# ============================================================================
# Integer / permutation boosts
# ============================================================================

def integer_boost(points):
    points = torch.as_tensor(points)

    N, D = points.shape
    device = points.device

    ranks = []

    for d in range(D):
        order = torch.argsort(points[:, d])

        rank = torch.empty(N, dtype=torch.long, device=device)
        rank[order] = torch.arange(N, dtype=torch.long, device=device)

        ranks.append(rank)

    return tuple(ranks)


def loop_boost(points):
    int_boost = integer_boost(points)

    N = int_boost[0].shape[0]
    device = points.device

    identity = torch.arange(N, dtype=torch.long, device=device)

    h_sources = (identity,) + int_boost
    h_targets = int_boost + (identity,)

    loop = tuple(
        _scatter_perm(h, hp, N, device)
        for h, hp in zip(h_sources, h_targets)
    )

    init_loop = loop[0]
    end_loop = loop[-1]

    circular_loop = list(loop[:-1])
    circular_loop[0] = init_loop[end_loop]

    return init_loop, tuple(circular_loop), end_loop


def _scatter_perm(h, hp, N, device):
    out = torch.zeros(N, dtype=torch.long, device=device)
    out[h] = hp
    return out


# ============================================================================
# Main algorithm
# ============================================================================

def torch_cartesian_sort(gridmap, points, max_iter=100, loop=None):

    g = torch.as_tensor(gridmap, dtype=torch.long)
    points = torch.as_tensor(points, device=g.device)

    dims = _active_dims(g)

    if len(dims) == 0:
        raise ValueError("gridmap has no active dimensions")

    if loop is None:
        loop = loop_boost(points[:, dims])

    init_loop, circular_loop, end_loop = loop

    g = init_loop[g]

    learning_curve = torch.zeros(max_iter + 1, dtype=torch.long, device=g.device)
    disorder = torch.tensor(1, dtype=torch.long, device=g.device)

    it = 0
    first_dim = dims[0]

    while (disorder > 0) and (it < max_iter):

        disorder = torch.tensor(0, dtype=torch.long, device=g.device)

        for k, d in enumerate(dims):

            heuristic = circular_loop[k]

            if not (it == 0 and d == first_dim):
                g = heuristic[g]

            disorder = disorder + (torch.diff(g, dim=d) < 0).sum()
            g = torch.sort(g, dim=d).values

        learning_curve[it] = disorder
        it += 1

    sorted_grid = end_loop[g]
    learning_curve = learning_curve[:it]

    return sorted_grid, learning_curve