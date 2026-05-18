import numpy as np

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