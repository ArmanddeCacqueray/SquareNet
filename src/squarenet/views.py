import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def localview(Xmap, wr, D, boundary="reflect", **pad_kwargs):
    """
    Compute local neighborhood views of a data Xmap 
    (related to the point cloud) on the latent grid.

    The function applies a sliding window over the grid to collect the local view

    Args:
        Xmap (np.ndarray): Input data of shape (*G, *C)

        D: dimension of the points 

        wr (int or sequence of int): window radius per grid axis.
            Window size per axis is: ws = 2 * wr + 1

        boundary (str): Padding mode = boundary condition passed 
            to `np.pad` (e.g. "constant", "reflect", "edge", ...)

        **pad_kwargs: additional arguments forwarded to np.pad, 
            see Numpy doc

    Returns:
        np.ndarray: Local neighborhood views of shape (*G, *C, *ws) 
    """
    #Entry point: Xmap (*G, *C)
    NC = len(Xmap.shape) - D
    assert NC>=0, "D should mmatch number of axis of the mapper"

    # 2. Normalize window radius, convert to window size
    if isinstance(wr, int):
        wr = (wr,) * D
    elif len(wr) != D:
        raise ValueError(f"wr length ({len(wr)}) must match grid dim ({D})")
    ws = tuple(2 * w + 1 for w in wr)

    # 3. Padding (for boundary conditions)
    pad_width = [(w, w) for w in wr] + [(0, 0)]*NC
    X_padded = np.pad(
        Xmap,
        pad_width=pad_width,
        mode=boundary,
        **pad_kwargs
    )

    # 4. Sliding window
    Xview = sliding_window_view(
        X_padded,
        window_shape=ws,
        axis=tuple(range(D))
    )
    
    return Xview #(*G, *C, *ws) 

def lazylocalview(*args, **kwargs):
    """
    wrapper on localview for cleverly requesting a (small) subset of the views.
    args[0] must be the selection mask which specify wich views are requested

    Returns:
        WindowCollector: dictionary key -> View[key]
    """
    selection = args[0]
    WindowCollector = {}
    Xview = localview(*args[1:], **kwargs)
    for sel in selection:
        WindowCollector[sel] = Xview[sel]
    return  WindowCollector # sel -> (*G[sel], *C, *ws)