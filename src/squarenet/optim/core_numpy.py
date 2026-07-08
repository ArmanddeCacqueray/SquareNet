import numpy as np
from ..utils import progress_bar
from .numpy.booster import loop_boost
from .numpy.subgrid import random_subgrid_split
from .numpy.hashtable import HashTable

""""
=============================================
cartesian sort - fast, robust and ultimate
=============================================
"""

def numpy_cartesian_sort(
    gridmap,
    points,
    method="fast",
    max_iter=100,
    verbose=2,
    loop=None,
    loopseq="decreasing"
):
    methods = {
        "fast": fast_cartesian_sort,
        "robust": robust_cartesian_sort,
        "ultimate": ultimate_cartesian_sort,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Expected one of {list(methods.keys())}"
        )

    return methods[method](
        gridmap,
        points,
        max_iter=max_iter,
        verbose=verbose,
        loop=loop,
        loopseq=loopseq,
    )

# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _prepare(gridmap, points, loop, loopseq):
    """Compute Dims (ordered), resolve loop, apply init_loop to g."""
    g = gridmap
    gshape = np.array(g.shape)
    Dims = np.where(gshape > 1)[0]

    if loopseq == "decreasing":
        Dims = Dims[np.argsort(-gshape[Dims])]
    elif loopseq == "random":
        Dims = np.random.permutation(Dims)
    else:
        raise ValueError(f"unknown loopseq {loopseq!r}, should be 'decreasing' or 'random'")

    if loop is None:
        loop = loop_boost(points[:, Dims])
    init_loop, circular_loop, end_loop = loop

    g = init_loop[g]
    return g, Dims, loop, circular_loop, end_loop


def _check_convergence(g, Dims):
    """Return total disorder count across all active dimensions."""
    return sum(int(np.sum(np.diff(g, axis=d) < 0)) for d in Dims)


def _cleanup(g, points, end_loop, loop, loopseq, max_iter, verbose):
    """Apply end_loop then run a final fast_cartesian_sort pass."""
    g = end_loop[g]
    g, lc = fast_cartesian_sort(
        g, points,
        max_iter=max_iter, verbose=verbose,
        loop=loop, loopseq=loopseq,
    )
    return np.ascontiguousarray(g), lc


def _log(verbose, it, max_iter, done=False):
    if verbose >= 2:
        progress_bar((max_iter - 1) if done else it % max_iter, max_iter)


# ─────────────────────────────────────────────
# Core sort loop (used by fast_cartesian_sort)
# ─────────────────────────────────────────────

def _run_sort_loop(g, Dims, circular_loop, max_iter, verbose, skip_first_heuristic=True):
    """
    Standard circular sort loop.
    Returns (g, learning_curve).
    skip_first_heuristic replicates the original `if not (it == 0 and d == Dims[0])` guard.
    """
    learning_curve = []

    for it in range(max_iter):
        _log(verbose, it, max_iter)
        disorder = 0

        for d_id, (d, heuristic) in enumerate(zip(Dims, circular_loop)):
            if not (skip_first_heuristic and it == 0 and d_id == 0):
                g = heuristic[g]

            disorder += _check_convergence(g, [d])
            g.sort(axis=d)

        learning_curve.append(disorder)
        if disorder == 0:
            _log(verbose, it, max_iter, done=True)
            break

    return g, learning_curve


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def fast_cartesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    """
    Args:
        gridmap (np.ndarray[int]): index map such that cloud_features[gridmap]
            writes any feature (N, *C) of the point-cloud (N, D) on a grid.
        max_iter (int): maximum number of iterations.
    Returns:
        gridmap: spatially coherent sorted index map.
        learning_curve (list): disorder per iteration; converges to 0.
    """
    g, Dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)
    g, learning_curve = _run_sort_loop(g, Dims, circular_loop, max_iter, verbose)

    g = end_loop[g]
    return np.ascontiguousarray(g), learning_curve


def robust_cartesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    """
    Robust variant: sorts only random independent subgrids each iteration to
    escape local minima, then finishes with a standard fast_cartesian_sort pass.

    Args / Returns: same as fast_cartesian_sort.
    """
    g, Dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)
    gshape = np.array(gridmap.shape)

    learning_curve = []
    circular = False

    for it in range(max_iter):
        _log(verbose, it, max_iter)
        disorder = 0

        subgrids = random_subgrid_split(gshape[Dims], Dims)

        for d_id, (d, heuristic) in enumerate(zip(Dims, circular_loop)):
            if circular:
                g = heuristic[g]
            circular = True

            disorder += _check_convergence(g, [d])

            for sub in subgrids[d_id]:
                gsub = g[sub]
                gsub.sort(axis=d)
                g[sub] = gsub

        learning_curve.append(disorder)
        if disorder == 0:
            _log(verbose, it, max_iter, done=True)
            break

    g, lc2 = _cleanup(g, points, end_loop, loop, loopseq, max_iter, verbose)
    if verbose >= 2:
        print("")

    return g, learning_curve + lc2


def ultimate_cartesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    """
    Ultimate variant: runs robust_cartesian_sort, then refines with a
    HashTable-based sort phase for 4×max_iter iterations.

    Args / Returns: same as fast_cartesian_sort.
    """
    g, Dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)

    # Phase 1 — robust sort
    g, learning_curve = robust_cartesian_sort(
        g, points,
        max_iter=max_iter, verbose=verbose,
        loop=loop, loopseq=loopseq,
    )

    # Phase 2 — HashTable refinement
    htable = HashTable(g, Dims)
    circular = False

    for it in range(4 * max_iter):
        _log(verbose, it, max_iter)
        for heuristic in circular_loop:
            if circular:
                htable.gtable = heuristic[htable.gtable]
            circular = True
            htable.sort()

    if verbose >= 2:
        print("")

    g, lc2 = _cleanup(htable.gtable, points, end_loop, loop, loopseq, max_iter, verbose)
    return g, learning_curve + lc2