import numpy as np

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