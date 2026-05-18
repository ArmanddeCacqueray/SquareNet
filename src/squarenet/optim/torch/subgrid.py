import torch
import numpy as np

def torch_ix_(*tensors):
    """PyTorch equivalent of np.ix_ for advanced orthogonal indexing."""
    ndim = len(tensors)
    out = []
    for c, t in enumerate(tensors):
        shape = [1] * ndim
        shape[c] = -1
        out.append(t.reshape(shape))
    return tuple(out)

def random_subgrid_split(gshape, Dims, device="cpu"):
    """
    PyTorch implementation of random_subgrid_split supporting CPU and GPU execution.
    """
    ndim = len(gshape)
    D = len(Dims)
    
    perms = [np.random.permutation(gshape[d]) for d in Dims]
    x0 = [np.sort(p[:len(p)//2]) for p in perms]
    x1 = [np.sort(p[len(p)//2:]) for p in perms]
    subsets = [x0, x1]

    subgrids = [[] for _ in range(D)]
    n_per_dim = 2**D // D
    
    d_ids = np.random.permutation(np.arange(2**D))[:D * n_per_dim].reshape(D, n_per_dim)

    # Pré-remplissage des dimensions inactives
    base_idx = [None] * ndim
    for c in range(ndim):
        if c not in Dims:
            base_idx[c] = torch.arange(gshape[c], dtype=torch.int32, device=device)

    for di in range(D):
        for dj in range(n_per_dim):
            dbit = d_ids[di, dj]
            idx = list(base_idx)
            
            for i in range(D):
                bit = (dbit >> i) & 1
                axis = Dims[i]
                idx[axis] = torch.tensor(subsets[bit][i], dtype=torch.int32, device=device)
            
            subgrids[di].append(torch_ix_(*idx))
            
    return subgrids