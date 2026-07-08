import numpy as np
from warnings import warn


import numpy as np

def index_identity(shape):
    """index_identy[i, j, k, ...] = [i, j, k, ...]"""
    return np.moveaxis(np.indices(shape), 0, -1)

def make_stencil(gridshape, dtype=float):
    """
    Create a grid stencil ordered by distance to the center of the grid.
    This is a helper for embedding an arbitrary set of `n_target` points into
    a smooth convex subset of a hyperrectangular lattice.

    Invalid positions are initialized with signed infinities, while valid
    positions can later be filled with the input data according to `fill_rank`.

    Parameters
    ----------
    gridshape : tuple of int
        The shape of the target grid.
    dtype : data-type, optional
        The desired data-type for the stencil array (default is float).

    Returns
    -------
    stencil: ndarray of shape (n, d)
        A stencil prefilled with  signed infinity at forbidden positions.
    fill_rank : ndarray of shape (n,)
        Ranking of the lattice points by increasing distance to the grid
        center. The positions satisfying ``fill_rank < n_target`` are the
        locations where the first ``n_target`` points should be inserted.
    """ 
    d = len(gridshape)
    epsilon = 1e-4 * np.arange(1, d + 1) / d

    cube = index_identity(gridshape).reshape(-1, d)
    cube = 2 * cube - cube.max(axis=0, keepdims=True)

    stencil = (2 * cube.astype(dtype) - 1) * np.inf
    cube_norm = np.linalg.norm(cube + epsilon, axis = -1)
    fill_rank = np.argsort(np.argsort(cube_norm))

    return stencil, fill_rank

def fill_in(data, stencil, fill_rank, xp):
    assert data.ndim == 2, f"Data must be (n, d), got shape {data.shape}"
    assert stencil.ndim == 2, f"Stencil must be (n, d), got shape {stencil.shape}"
    
    n_target, d = data.shape
    
    assert stencil.shape[1] == d, (
        f"Dimension mismatch: data features ({d}) do not match stencil features ({stencil.shape[1]})."
    )
    
    fill_mask = fill_rank < n_target
    
    if xp.__name__ == "torch":
        full_data = stencil.clone()
        full_data[fill_mask] = data
        
    elif "jax" in xp.__name__:
        full_data = stencil.at[fill_mask].set(data)
        
    else:
        full_data = stencil.copy()
        full_data[fill_mask] = data
        
    return full_data

def dualgrid(grid, xp, N, IJ, D):
    # torch
    if xp.__name__ == "torch":
        identity = xp.stack(
            xp.meshgrid(
                *[
                    xp.arange(
                        s,
                        dtype=grid.dtype,
                        device=grid.device,
                    )
                    for s in IJ
                ],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(N, D)

        out = xp.empty(
            (N, D),
            dtype=grid.dtype,
            device=grid.device,
        )

        out[grid.reshape(-1)] = identity
        return out

    # numpy / jax
    identity = xp.stack(
        xp.meshgrid(
            *[xp.arange(s, dtype=grid.dtype) for s in IJ],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(N, D)

    # jax
    if xp.__name__.startswith("jax"):
        out = xp.zeros((N, D), dtype=grid.dtype)
        return out.at[grid.reshape(-1)].set(identity)

    # numpy
    out = xp.zeros((N, D), dtype=grid.dtype)
    out[grid.reshape(-1)] = identity
    return out


def dualgridflat(grid, xp, N):
    gr = grid.reshape(-1)

    # torch
    if xp.__name__ == "torch":
        identity = xp.arange(
            N,
            dtype=grid.dtype,
            device=grid.device,
        )

        out = xp.empty(
            N,
            dtype=grid.dtype,
            device=grid.device,
        )

        out[gr] = identity
        return out
    
    # numpy / jax
    identity = xp.arange(N, dtype=grid.dtype)

    # jax
    if xp.__name__.startswith("jax"):
        out = xp.zeros(N, dtype=grid.dtype)
        return out.at[gr].set(identity)

    # numpy
    out = xp.zeros(N, dtype=grid.dtype)
    out[gr] = identity
    return out  

def breakpoint():
    raise RuntimeError("STOP checkpoint.\n Everything allright...")

def project(gridpoints, feature_axes=(0, 1), index=0):
    grid_ndim = gridpoints.ndim - 1
    
    selection = [index] * grid_ndim

    axes = np.arange(grid_ndim)
    for ax in axes:
        if ax in feature_axes:
            selection[ax] = slice(None)
        else:
            selection[ax] = gridpoints.shape[ax]//2
    
    x = gridpoints[tuple(selection)]
    
    current_order = sorted(range(len(feature_axes)), key=lambda i: feature_axes[i])
    new_order = np.argsort(current_order)
    
    x = x.transpose(list(new_order) + [len(feature_axes)])
    x = x[..., list(feature_axes)]
    
    return x

def progress_bar(it, total, bar_length=30):
    progress = it / total
    filled = int(bar_length * progress)
    bar = "█" * filled + "-" * max(0,(bar_length - filled-1))

    if it >= total-1:
        print(f"\r[{bar}] {total}/{total}")
    else:
        print(f"\r[{bar}] {it}/{total}", end="")

def printmatrix(arr):
    max_x = arr.max(axis=0)
    max_y = arr.max(axis=1)

    max_x = np.maximum(max_x, max_x[::-1])
    max_y = np.maximum(max_y, max_y[::-1])

    x_idx = np.where(max_x >= 0)[0]
    y_idx = np.where(max_y >= 0)[0]

    if len(x_idx) == 0 or len(y_idx) == 0:
        arr = np.zeros((1, 1))
    else:
        x0, x1 = x_idx[0], x_idx[-1]
        y0, y1 = y_idx[0], y_idx[-1]

        arr = arr[y0:y1+1, x0:x1+1]

    width = max(len(str(x)) for x in arr.flatten())

    hx, hy  = arr.shape
    wrx, wry = hx // 2, hy//2  # center

    marker = f"{'■':>{width}}"

    for i, row in enumerate(arr):
        line = []
        for j, x in enumerate(row):
            if i == wrx and j == wry:
                line.append(marker)
            else:
                line.append(f"{x:{width}d}" if x >= 0 else " " * width)

        print(" ".join(line))
  

def show_search_result(left, right, true, points, sn):
    import matplotlib.pyplot as plt
    print("true index", sn.mapidx(true))
    print("with search sorted:", left, right)

    point_found_l = points[sn.invert_mapidx(left)]
    point_found_r = points[sn.invert_mapidx(right)]
    plt.figure(figsize = (6, 6))
    plt.scatter(points[:, 0], points[:, 1], color = "grey", s = 10000/len(points))

    plt.scatter(point_found_l[0], point_found_l[1], s= 200, alpha = 0.5, color = "blue", label =  "found left")
    plt.scatter(point_found_r[0], point_found_r[1], s= 200, alpha = 0.5, color = "green", label =  "found right")
    plt.scatter(points[true][0], points[true][1], s= 200, marker = "x", color = "red", label =  "true")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc = "upper right")
    plt.show()