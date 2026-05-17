import numpy as np
from .core_numpy import np_carthesian_sort, integer_boost, loop_boost
from ..utils import progress_bar, from_backend, to_backend

def tangled_carthesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, backend = "numpy", loopseq="decreasing"):
    """
    Iteratively sort a gridmap along its Cartesian dimensions to minimise spatial disorder.

    Uses a point cloud to guide sorting, alternating between standard axis sorts and
    a pivot-based tangle/untangle phase to resolve non-local topological conflicts.
    Returns the sorted gridmap and the per-iteration disorder history.
    """
    device = "cpu"
    if hasattr(gridmap, "device"):
        device = gridmap.device
    g = from_backend(gridmap)
    pts = from_backend(points)
    gshape = np.array(g.shape)
    Dims = np.where(gshape > 1)[0]

    if loopseq == "decreasing":
        Dims = Dims[np.argsort(-gshape[Dims])]
    elif loopseq == "random":
        Dims = np.random.permutation(Dims)
    else:
        raise ValueError(f"Unknown loopseq '{loopseq}', expected 'decreasing' or 'random'.")
        
    init_loop, circular_loop, end_loop = loop_boost(points[:, Dims]) if loop is None else loop
    g = init_loop[g]

    for it in range(1000):
        if verbose >= 2:
            progress_bar(it % 100, 100)
            
        for d, heuristic in zip(Dims, circular_loop):
            if not (it == 0 and d == Dims[0]):
                g = heuristic[g]
                
            # Pivot phase: interleave axes, sort, restore
            dpivot = np.random.choice([ax for ax in Dims if ax != d])
            sl, antisl = make_even_pivot(g, dpivot)
            gpivot = tangle_pivot(g[sl], d, dpivot)
            gpivot.sort(axis=-1)
            g[sl] = untangle_pivot(gpivot, d, dpivot)

            for asl in antisl:
                g[asl].sort(axis=d)

    # last cleanup:
    g = end_loop[g]
    g, learning_curve2 = np_carthesian_sort(g, pts, max_iter=max_iter, verbose = verbose, 
                           loop = (init_loop, circular_loop, end_loop ), loopseq = loopseq)
    gridmap = to_backend(g, backend = backend, device = device, warnings_ = verbose >=1)
    return gridmap, learning_curve2


def make_even_pivot(g, dpivot):
    """
    Build slices that isolate a sub-grid with even size along `dpivot`.

    Returns a main slice covering the even-length region and a list of
    boundary slices for the leftover odd row(s), ensuring full coverage.
    """
    n = g.shape[dpivot]
    base = [slice(None)] * g.ndim

    if n % 2 == 0:
        choices = [slice(None), slice(1, -1)] if n >= 4 else [slice(None)]
    else:
        choices = [slice(1, None), slice(None, -1)]
    pivot_slice = choices[np.random.randint(len(choices))]

    sl = list(base)
    sl[dpivot] = pivot_slice

    antisl = []
    if pivot_slice == slice(1, -1):
        s0, s1 = list(base), list(base)
        s0[dpivot], s1[dpivot] = slice(0, 1), slice(-1, None)
        antisl = [tuple(s0), tuple(s1)]
    elif pivot_slice == slice(1, None):
        s0 = list(base); s0[dpivot] = slice(0, 1)
        antisl = [tuple(s0)]
    elif pivot_slice == slice(None, -1):
        s1 = list(base); s1[dpivot] = slice(-1, None)
        antisl = [tuple(s1)]

    return tuple(sl), antisl


def tangle_pivot(g, d, dpivot):
    """
    Interleave the pivot and sort axes into a single extended sort axis.

    Moves `dpivot` to -2 and `d` to -1, then reshapes to [..., Sp//2, 2*Sd]
    so that a single .sort(axis=-1) performs non-local cross-pivot sorting.
    Returns the transformed array and the metadata needed to invert the operation.
    """
    offset = 1 if d > dpivot else 0
    Sp = g.shape[dpivot]
    Sd = g.shape[d]

    # Bring working axes to the back: dpivot → -2, then d → -1
    g = np.moveaxis(g, dpivot, -1)
    g = np.moveaxis(g,  d-offset, -1)
    # Interleave: split pivot pairs, zip with sort axis, then flatten
    s = list(g.shape)
    g = g.reshape(s[:-2] + [Sp // 2, 2, Sd])
    g = g.swapaxes(-2, -1)                      # [..., Sp//2, Sd, 2]
    g = g.reshape(s[:-2] + [Sp // 2, 2 * Sd])   # sort axis is now -1

    return g


def untangle_pivot(g, d, dpivot):
    """
    Exact inverse of `tangle_pivot`: restore the original axis layout.
    """
    offset = 1 if d > dpivot else 0
    Sp, Sd    = g.shape[-2]*2, g.shape[-1]//2
    s = list(g.shape)
    g = g.reshape(s[:-2] + [Sp//2, Sd, 2])             # undo flatten
    g = g.swapaxes(-2, -1)                      # undo zip  → [..., Sp//2, 2, Sd]
    g = g.reshape(s[:-2] + [Sp, Sd])            # undo pair-split
    g = np.moveaxis(g,  -1, d-offset)
    g = np.moveaxis(g, -1, dpivot)
    return g
