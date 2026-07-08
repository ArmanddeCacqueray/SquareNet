"""
PyTorch implementation of Cartesian sort
GPU compatible with Chunked (Big Step) Optimization
"""

import torch
from .torch.booster import loop_boost
from .torch.subgrid import random_subgrid_split
from .torch.hashtable import HashTable


# ============================================================================
# Public API
# ============================================================================

def torch_cartesian_sort(gridmap, points, method="fast", max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    """PyTorch Cartesian sorting supporting CPU/GPU (Multi-method Dispatcher)."""
    if verbose >= 2:
        print(f"torch working ({method}) ...")

    methods = {
        "fast": fast_cartesian_sort,
        "robust": robust_cartesian_sort,
        "ultimate": ultimate_cartesian_sort,
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Expected one of {list(methods.keys())}")

    return methods[method](
        gridmap, points, max_iter=max_iter, verbose=verbose, loop=loop, loopseq=loopseq
    )

# ============================================================================
# Helpers
# ============================================================================

def _prepare(gridmap, points, loop, loopseq):
    g = torch.as_tensor(gridmap, dtype=torch.int32)
    points = torch.as_tensor(points, device=g.device)
    shape = list(g.shape)
    
    dims = [i for i, s in enumerate(shape) if s > 1]

    if loopseq == "decreasing":
        dims.sort(key=lambda d: -shape[d])
    elif loopseq == "random":
        import random
        random.shuffle(dims)
    else:
        raise ValueError(f"unknown loopseq {loopseq!r}, should be 'decreasing' or 'random'")

    dims = tuple(dims)

    if loop is None:
        loop = loop_boost(points[:, dims])
    init_loop, circular_loop, end_loop = loop

    g = init_loop[g]
    return g, dims, loop, circular_loop, end_loop


def _check_convergence(g, dims, circular_loop):
    disorder = 0
    for d_id, (d, heuristic) in enumerate(zip(dims, circular_loop)):
                g = heuristic[g]
                disorder += (torch.diff(g, dim=d) < 0).sum().item()

    # .item() causes a CPU-GPU sync. Call this sparingly!
    return disorder


def _cleanup(g, points, end_loop, loop, loopseq, max_iter, verbose):
    g = end_loop[g]
    g, lc = fast_cartesian_sort(
        g, points,
        max_iter=max_iter, verbose=verbose,
        loop=loop, loopseq=loopseq,
    )
    return g.contiguous(), lc


# ============================================================================
# Core Sort Implementations
# ============================================================================

def fast_cartesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    g, dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)

    learning_curve = []
    chunk_size = 10
    first_dim = dims[0]

    # Warmup: Handle the very first step manually to avoid 'if' conditions in the loop
    for k, d in enumerate(dims):
        if d != first_dim:
            g = circular_loop[k][g]
        g = torch.sort(g, dim=d).values

    for _ in range(max_iter // chunk_size):
        # 10 blind unrolled iterations
        for _ in range(chunk_size):
            for k, d in enumerate(dims):
                g = circular_loop[k][g]
                g = torch.sort(g, dim=d).values

        # Single convergence check per chunk
        disorder = _check_convergence(g, dims, circular_loop)
        learning_curve.extend([disorder] * 10)
        if disorder == 0:
            break

    sorted_grid = end_loop[g]
    return sorted_grid, learning_curve


def robust_cartesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    g, dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)
    gshape = list(gridmap.shape)

    learning_curve = []
    circular = False
    chunk_size = 10

    for big_it in range(max_iter // chunk_size):
        # 10 blind unrolled inner steps
        for _ in range(chunk_size):
            subgrids = random_subgrid_split(gshape, dims, device=g.device)

            for d_id, (d, heuristic) in enumerate(zip(dims, circular_loop)):
                if circular:
                    g = heuristic[g]
                circular = True

                for sub in subgrids[d_id]:
                    g[sub] = torch.sort(g[sub], dim=d).values

        # Check convergence only once per chunk
        disorder = _check_convergence(g, dims, circular_loop)
        learning_curve.extend([disorder] * 10)
        if disorder == 0:
            break

    g, lc2 = _cleanup(g, points, end_loop, loop, loopseq, max_iter, verbose)
    learning_curve.extend(lc2)
    return g, learning_curve


def ultimate_cartesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    g, dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)

    # Phase 1 — Robust sort
    g, lc1 = robust_cartesian_sort(
        g, points,
        max_iter=max_iter, verbose=verbose,
        loop=loop, loopseq=loopseq,
    )

    # Phase 2 — HashTable refinement with big steps
    htable = HashTable(g, dims)
    circular = False
    chunk_size = 10

    for _ in range((4 * max_iter) // chunk_size):
        for _ in range(chunk_size):
            for heuristic in circular_loop:
                if circular:
                    htable.gtable = heuristic[htable.gtable]
                circular = True
                htable.sort()

    g, lc2 = _cleanup(htable.gtable, points, end_loop, loop, loopseq, max_iter, verbose)
    lc1.extend(lc2)
    return g, lc1