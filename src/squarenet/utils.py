import numpy as np
from warnings import warn

def index_identity(shape):
     #index_identity[i,j,k] = [i, j, k]
     return np.moveaxis(np.indices(shape), 0, -1)

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
    for i, ax in enumerate(feature_axes):
        selection[ax] = slice(None)
    
    x = gridpoints[tuple(selection)]
    
    current_order = sorted(range(len(feature_axes)), key=lambda i: feature_axes[i])
    new_order = np.argsort(current_order)
    
    x = x.transpose(list(new_order) + [len(feature_axes)])
    x = x[..., list(feature_axes)]
    
    return x


def from_backend(x):
    """
    Convert torch/jax/numpy array to numpy safely.
    """
    if isinstance(x, np.ndarray):
        return x

    # torch
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()

    # jax
    if hasattr(x, "__array__"):
        return np.asarray(x)

    return np.asarray(x)

def to_backend(x, backend="numpy", device = "cpu", warnings_ = True):
    """
    Convert numpy array to target backend.
    """
    if backend == "numpy":
        return np.asarray(x)

    if backend == "torch":
        import torch
        device = torch.device(device)
        x_device = getattr(x, "device", None)
        if (
            warnings_
            and x_device is not None
            and hasattr(x_device, "type")
            and x_device.type == "cpu"
            and device.type != "cpu"
        ):
            warn(
                "Downgrading tensor as initial device was GPU but asked device is CPU"
            )

        return torch.as_tensor(x, device=device)

    if backend == "jax":
        import jax.numpy as jnp
        return jnp.asarray(x)

    raise ValueError(f"Unknown backend: {backend}")

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