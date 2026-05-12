import numpy as np
from ..utils import progress_bar

""""
=============================================
=============================================
Numpy version of carthesian sort,
with a boosted circular loop
=============================================
=============================================
"""

def np_carthesian_sort(gridmap, points, max_iter=100, verbose = 2, loop = None, loopseq = "decreasing"):
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
    g = gridmap
    gshape = np.array(g.shape)
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
    init_loop, circular_loop, end_loop = loop_boost(points[:, Dims]) if loop is None else loop
    g = init_loop[g]
    
    for it in range(max_iter):
        if verbose >= 2:
            progress_bar((2*it)%100, 100)
        disorder = 0
        for d, heuristic in zip(Dims, circular_loop):
            if it+d>0:
                g = heuristic[g]
            
            # --- 1. Check for convergence ---
            diff = np.diff(g, axis=d)
            disorder += np.sum(diff < 0)

            # --- 2. Sorting Phase ---
            g.sort(axis = d)

        learning_curve.append(disorder)           
        if disorder == 0:
            if verbose >=2:
                progress_bar(99, 100)
            break

    # last cleanup:
    g = end_loop[g]
    gridmap = np.ascontiguousarray(g)
    return gridmap, learning_curve
# ============================================
# ============================================
# Boosters
# ============================================
# ============================================
def integer_boost(points):
    """
    Booster: Convert points to integer 
    to boost sort_increasing function

    Args:
        - points (np.ndarray) (N,D): point cloud

    Returns:
        - h_int (list of np.uint32 arrays): points as integers
    """
    N, D = points.shape
    h_int = []
    for d in range(D):
        order = np.argsort(points[:, d])      
        ranks = np.empty(N, dtype=np.int32)
        ranks[order] = np.arange(N)
        h_int.append(ranks)
    return h_int

def loop_boost(points):
    """
    Booster: make looping over axis a bit faster

    Args:
        - points (np.ndarray) (N,D): point cloud
    Return:
        - loop (...): list of permutations such that 
        loop[d](h_int[d][n]) = h_int[d+1][n]
        - end_loop (...): permutation such that 
        end_loop(h_int[-1][n]) = n 
    """
    int_boost = integer_boost(points)
    N = len(int_boost[0])
    identity = np.arange(N, dtype=np.int32)
    #start the loop
    h_int =  [identity] + int_boost
    #close the loop
    h_int_plus = int_boost + [identity]
    loop = []

    for h, hplus in zip(h_int, h_int_plus):
        sigma = np.zeros(N, dtype=np.int32)
        sigma[h] = hplus
        loop.append(np.ascontiguousarray(sigma))
        
    init_loop, end_loop = loop[0], loop[-1]
    circular_loop = loop[:-1]
    circular_loop[0] = init_loop[end_loop]
    return init_loop, circular_loop, end_loop