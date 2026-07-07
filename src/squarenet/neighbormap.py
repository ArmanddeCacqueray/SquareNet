import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def distfunction(x, y, axis):
    return ((x-y)**2).sum(axis = axis)

def distkernel(x, xview):
    d = len(x.shape) - 1
    return distfunction(x[..., None, None], xview, axis = d)

def neighbormap(grided_points, mapidx, criterion = "rank", thresholdcut = 1, 
                kernel = distkernel, windowradius= 10, projection = (0,1), sample_size = 1_000_000):
    gpts = np.asarray(grided_points)
    d = len(gpts.shape) - 1
    wr = windowradius
    ws = 2*wr+1
    N = np.prod(gpts.shape[:-1])
    D = gpts.shape[-1]

    selection = None

    if N*D*(ws**2) >= sample_size:
        sample_size_x = min(sample_size//(ws**2), N)
        selection = np.random.choice(N, replace = False, size = sample_size_x)
        selection = mapidx(selection)
    else:
        selection = mapidx(np.arange(N))
    mask = np.isfinite(grided_points[selection]).all(axis=-1)
    selection = tuple(dim_coords[mask] for dim_coords in selection)
        
    pad_width = [(0, 0)]*(d+1)
    for gax in projection:
        pad_width[gax] = (wr, wr)
    gpad = np.pad(
        gpts,
        pad_width=pad_width,
        mode="constant",
        constant_values = 1_234_567
    )

    gview = sliding_window_view(
        gpad,
        window_shape=(ws, ws),
        axis=projection
    )

    dists = kernel(gpts[selection], gview[selection])

    if criterion =="value":
        hotspots = (dists <= thresholdcut).sum(axis = tuple(range(d)))
    if criterion == "rank":
        dists = dists.reshape(-1, ws*ws)
        ranks = np.argsort(np.argsort(dists, axis = -1), axis = -1) - 1
        hotspots = (ranks <= thresholdcut).sum(axis = 0).reshape(ws, ws)
    hotspots[wr, wr] = 0
    return  hotspots
