import numpy as np
from ..utils import progress_bar, from_backend, to_backend
from .core_numpy import np_carthesian_sort, integer_boost, loop_boost

""""
=============================================
=============================================
Robust version of carthesian sort, kind of
interpolating a pool of solutions (by sorting
only subgrids) to avoid 
geting stuck on a local minima.

Native support only  Numpy for now
=============================================
=============================================
"""

def random_subgrid_split(gshape, Dims):
        # random partition of each axis

    D = len(Dims)
    perms = [
        np.random.permutation(gshape[d])
        for d in Dims
    ]

    x0 = [
        np.sort(p[:len(p)//2])
        for p in perms
    ]

    x1 = [
        np.sort(p[len(p)//2:])
        for p in perms
    ]

    subsets = [x0, x1]

    # build one subgrid per dimension d
    subgrids = [[] for _ in range(D)]

    n_per_dim = 2**D//D

    d_ids = np.random.permutation(np.arange(2**D))[:D*n_per_dim].reshape(D, n_per_dim)

    for di in range(D):
        for dj in range(n_per_dim):
            # category encoded as binary bits
            dbit = d_ids[di, dj] 

            idx = []

            for i in range(D):

                bit = (dbit >> i) & 1

                idx.append(
                    subsets[bit][i]
                )

            subgrids[di].append(np.ix_(*idx))
    return subgrids

def robust_carthesian_sort(gridmap, points, max_iter=100, verbose = 2, 
                           loop = None, loopseq = "decreasing", backend = "numpy"):
    """
    Args: 
        -gridmap (np.array of ints):
        an initial gridmap such that cloud_features[gridmap] 
        write any feature (N, *C) of the point-cloud (N, D) 
        on a grid (N1, ..., ND, *C)

        -max_iter (int):
        last step after which algorithm shall stop
        even if it hasn't converged yet
    Returns:
        -gridmap
        sorted gridmap such that cloud_features[gridmap]
        now write the feature on a spatially coherent grid
        -learningcurve (list of values):
        track the performance of the optimisation process.
        should converge to 0
    """
    device = "cpu"
    if hasattr(gridmap, "device"):
        device = gridmap.device
    g = from_backend(gridmap)
    pts = from_backend(points)
    gshape = np.array(g.shape)
    D = len(gshape)
    Dims = np.where(gshape > 1)[0] #exclude manifold dimensions

    if loopseq == "decreasing":
        Dims = Dims[np.argsort(-gshape[Dims])]
    elif loopseq == "random":
        Dims = np.random.permute(Dims)
    else:
        raise(f"unknown loopseq {loopseq}, should be 'decreasing'or 'random'")


    learning_curve = []

    #init_loop: index to heuristic 0
    #loop[d+1]: heuristic d to heuristic (d+1)%D
    #end_loop: heuristic D-1 to index
    init_loop, circular_loop, end_loop = loop_boost(pts[:, Dims]) if loop is None else loop
    g = init_loop[g]

    for it in range(100):
        if verbose >= 2:
            progress_bar((2*it)%100, 100)
        disorder = 0

        subgrids = random_subgrid_split(gshape[Dims], Dims)

        for d_id, (d, heuristic) in enumerate(zip(Dims, circular_loop)):
            if it+d>0:
                g = heuristic[g]
            
            # --- 1. Check for convergence ---
            diff = np.diff(g, axis=d)
            disorder += np.sum(diff < 0)

            # --- 2. Sorting Phase ---
            #only on independant subgrids
            for sub in subgrids[d_id]:
                gsub = g[sub]
                gsub.sort(axis = d)
                g[sub] = gsub 

        learning_curve.append(disorder)           
        if disorder == 0:
            if verbose >=2:
                progress_bar(99, 100)
            break

    # last cleanup:
    g = end_loop[g]
    g, learning_curve2 = np_carthesian_sort(g, pts, max_iter=max_iter, verbose = verbose, 
                           loop = (init_loop, circular_loop, end_loop ), loopseq = loopseq)
    gridmap = to_backend(g, backend = backend, device = device, warnings_ = verbose >=1)
    return gridmap, learning_curve + learning_curve2
