import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from .utils import project

_STYLES = ["checkerboard", "mesh", "scatter"]

_CONFIG_DESCR = """\
===============================================================
I. MAIN ARGUMENTS  —  passed directly to sqplot()
===============================================================
    style      : rendering style. One of:
                    'checkerboard' — points coloured in a 2-colour tile pattern
                    'mesh'         — grid lines drawn at 4 levels of density,
                    'scatter'      — plain point cloud; depth-coloured (cmap) in
                                    3-D, black in 2-D.
    animate    : False → single static PNG.
                True  → GIF that morphs continuously from the input grid
                        to the identity grid and back.
    save       : whether to write the output to disk.
    save_path  : destination path.

===============================================================
II. EXTRA ARGUMENTS  —  passed as cfg = {...}
===============================================================
All keys are optional. Only specify what you want to override;
everything else is filled from the defaults shown here.

Layout
    figsize      : (width, heigth) size of the base figure

Projection
    projection   : 3-tuple of axis indices, e.g. (0, 1, 2).
                    Used if ndim >= 3. this selects both 
                    which axes to keep AND in what order

Scale & density
    scale_factor : positive integer. bigger -> finer details
    pointsize    : base size for a single point 
    linewidth    : base width for a mesh lines.
    mesh_long_edge : length ratio (relative to the median) above wich an edge will not be ploted in 'mesh' style

Colours
    colors_checkerboard       : 2 colors for the 'checkerboard' style.
    cmap_scatter         : colormap e.g. 'plasma', 'coolwarm' used in 'scatter' to encode the depth.

Animation
    frames       : number of frames in the animation
    interval     : inter-frame delay in milliseconds (for interactive sessions).
    fps          : frames per second written (for the saved GIF).

Display
    show         : if True, calls plt.show(). 
                Will fail if session is not interactive
\n
"""

# =========================================================
# UTILS
# =========================================================
def default_config():
    return {
        "supported styles": _STYLES,
        # --- layout ---
        "figsize"   : (4, 4),
        # --- projection ---
        "projection" : (0, 1, 2),
        # --- scale ---
        "scale_factor" : 1,
        "pointsize"      : 1,
        "linewidth" : 1, #mesh style only
        "mesh_long_edge": 30, #mesh only

        # --- style ---
        "colors_checkerboard"    : ("peru", "mediumblue"), #checkerboard style only
        "cmap_scatter"      : "plasma",        #scatter style only (3-D depth)
        # --- animation ---
        "frames"    : 60,
        "interval"  : 30,
        "fps"       : 20,
        # --- show and export ---
        "show" : True,
    }

default_config.__doc__ = _CONFIG_DESCR

def kill_long_edges(x, threshold = 1):
    long_edge = (np.diff(x, axis = 0)**2).sum(axis = -1) >= threshold
    x[:-1][long_edge] = np.nan
    

def validate(g):
    g = np.asarray(g.astype(float)).copy()
    g[~np.isfinite(g)] = np.nan
    return g

def atmost3D(g, projection=(0, 1, 2)):
    if g.ndim <= 3:
        return g
    return project(g, feature_axes=projection)

def _safe_path(path: str, suffix: str) -> Path:
    p = Path(path)
    if p.suffix != suffix:
        p = p.with_suffix(suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return p
    stem, parent = p.stem, p.parent
    i = 1
    while (candidate := parent / f"{stem}_{i}{suffix}").exists():
        i += 1
    return candidate


# =========================================================
# GRID PREPROCESSING
# =========================================================

def _prepare_grid(g, DS=1):
    n1, n2 = g.shape[:2]
    n3 = 1
    is3D = ((g.ndim -1) == 3)
    if is3D:
        n3 = g.shape[2]
    npoints = (n1*n2*n3)
    if min(g.shape[:-1]) <= 10: #downsampling (thin case)
        if npoints > 100_000:
            DS = max(DS, 2)   
        if npoints > 300_000:
            DS = max(DS, 3)
        if npoints > 1_000_000:
            DS = max(DS, 4)
        if npoints > 3_000_000:
            DS = max(DS, 5)

    else: #downsampling (volumic case)
        if npoints > 500_000:
            DS = max(DS, 2)
        if npoints > 2_000_000:
             DS = max(DS, 3)
        if npoints > 5_000_000:
            DS = max(DS, 4)


    safe_mean_g = np.nanmean(g.reshape(-1, g.shape[-1]), axis = 0)

    if is3D:
        g = g[::DS, ::DS, ::DS, :3]
        P = np.array([
            [ 1,  1 / np.sqrt(3), -np.sqrt(2)/ np.sqrt(3)],
            [-1,  1 / np.sqrt(3),  -np.sqrt(2)/ np.sqrt(3)],
            [ 0,  2 / np.sqrt(3), np.sqrt(2)/ np.sqrt(3)],
        ])
        g = (g - safe_mean_g[None, None, None, :]) @ P
    else:
        g = g[::DS, ::DS, :2]
        g = g - safe_mean_g[None, None, :]

    g_min, g_max = np.nanmin(g), np.nanmax(g)

    g = (g - g_min) / (g_max - g_min + 1e-8)
    return g


# =========================================================
# SURFACE EXTRACTION
# =========================================================

def _get_surfaces(grid):
    D = grid.shape[-1]
    if grid.ndim == 4:
        surfaces = [grid[0, :, :], grid[:, 0, :], grid[:, :, -1]]
        shell_mask = np.ones_like(grid, dtype = bool)
        i, j, k, _ = grid.shape
        mid_i, mid_j, mid_k = i//2, j//2, k//2
        di, dj, dk = max(i-mid_i-10, 0), max(j-mid_j-10, 0), max(k-mid_k-10, 0)
        shell_mask[mid_i-di:mid_i+di, mid_j-dj: mid_j+dj, mid_k-dk:mid_k+dk] = False
        shell_mask = shell_mask.reshape(-1, D)[:, 0]
        return surfaces, grid.reshape(-1, D)[shell_mask]
    return [grid], grid.reshape(-1, D)


# =========================================================
# CHECKERBOARD MASK
# =========================================================

def _checkerboard_mask(surface, stepi, stepj):
    ni, nj = surface.shape[:2]
    ii, jj = np.indices((ni, nj))
    return (((ii // stepi) % 2) == ((jj // stepj) % 2)).reshape(-1)


# =========================================================
# CORE — BUILD ANIMATION
# =========================================================
def _build_animation(grid, cfg):
    """
    Layout depends on style:
    - "checkerboard" : tile the surface (checkerboard -like).
    - "mesh"         : mesh the surface (horizontal and vertical lines).
    - "scatter"      : scatter the surface (depth map if ndim >= 3).
    """
    sf = cfg["scale_factor"]
    style = cfg["style"] 
    anim = cfg["animate"]
    assert style in ["mesh", "checkerboard", "scatter"], \
        f"Unknown plot style {style}, must be 'mesh', 'checkerboard' or 'scatter'"

    if anim and (style == "checkerboard"):
        scales = [8]
        lws = [-1]
    elif (style == "checkerboard"):
        scales = [2, 4, 8]
        lws = [-1, -1, -1]
    elif (style == "mesh"):
        scales = [2, 4, 8, 16]
        lws = [1.5, 1, 0.7, 0.4]
        lws = [l*cfg["linewidth"] for l in lws]
    else:  # scatter — scales/lws unused but keep variables defined
        scales = [1]
        lws = [-1]

    scales = [sf*sc for sc in scales]
    maxscales = max(scales)
    DS      = 1 
    frames  = cfg["frames"]
    colors  = cfg["colors_checkerboard"]
    figsize = cfg["figsize"]
    cmap    = cfg["cmap_scatter"]

    # --- Precompute both endpoints ---
    coords       = [np.linspace(0, 1, s) for s in grid.shape[:-1]]
    identity_raw = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)

    g_prep  = _prepare_grid(grid, DS=DS)
    id_prep = _prepare_grid(identity_raw, DS=DS)

    is_3d   = g_prep.ndim == 4

    # --- Point size normalisation ---
    n1, n2  = g_prep.shape[:2]
    n3      = g_prep.shape[2] if is_3d else 0
    npoints = n1 * n2 + (n1 * n3 + n2 * n3 if is_3d else 0)
    pt_size = cfg["pointsize"] * (80_000 / npoints)

    def interp(t):
        return (1 - t) * g_prep + t * id_prep

    # -------------------------------------------------------
    # Build artists at t=0/1
    # -------------------------------------------------------
    surfaces0, full0 = _get_surfaces(interp(0.0))
    surfaces1, full1 = _get_surfaces(interp(1.0))

    if style == "mesh":
        long_edge = 0
        axes = [0, 1, 2] if n3 > 1 else [0, 1]
        for axis in axes:
            long_edge += np.nanmedian((np.diff(g_prep, axis = axis)**2).sum(axis = -1))
        long_edge *= ((cfg["mesh_long_edge"])**2)/len(axes)


    if is_3d:
        depth = full0[:, 2]
        order  = np.argsort(depth)
        full0, full1 = np.ascontiguousarray(full0[order]), np.ascontiguousarray(full1[order])
        if style == "scatter":
            for i in range(len(surfaces0)):
                D = surfaces0[i].shape[-1]
                s0, s1 = surfaces0[i].reshape(-1, D), surfaces1[i].reshape(-1, D)
                depth = s0[:, 2]
                order  = np.argsort(depth)
                surfaces0[i], surfaces1[i] = s0[order], s1[order]

    def _lerp(t):
        return [(1 - t) * s0 + t * s1 for (s0, s1) in zip(surfaces0, surfaces1)], (1 - t) * full0 + t * full1


    # -------------------------------------------------------
    # Figure setup
    # -------------------------------------------------------
    if style == "checkerboard":
        fig, axes = plt.subplots(
            1, len(scales),
            figsize=(figsize[0] * len(scales), figsize[1])
        )
        axes = [axes] if len(scales) == 1 else list(axes)
        for ax in axes:
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.axis("off")
            ax.set_aspect("equal")
    elif style == "scatter":
        fig, ax_single = plt.subplots(1, 1, figsize=figsize)
        ax_single.set_xlim(-0.05, 1.05)
        ax_single.set_ylim(-0.05, 1.05)
        ax_single.axis("off")
        ax_single.set_aspect("equal")
        axes = [ax_single]
    else:
        fig, ax_single = plt.subplots(1, 1, figsize=figsize)
        ax_single.set_xlim(-0.05, 1.05)
        ax_single.set_ylim(-0.05, 1.05)
        ax_single.axis("off")
        ax_single.set_aspect("equal")
        axes = [ax_single] * len(scales)
    # -------------------------------------------------------
    # checkerboard / mesh  (unchanged logic below)
    # -------------------------------------------------------
    seen_axes = []
    bg_artists = []
    for ax in axes:
        if id(ax) not in seen_axes:
            color = "lightgrey"
            if (style == "scatter") and (not is_3d):
                color = "black"
            seen_axes.append(id(ax))
            bg_artists.append(
                ax.scatter(
                    full0[:, 0], full0[:, 1],
                    c=color, s=pt_size, linewidths=0
                )
            )

    scale_artists = []

    for ax, sc, lw in zip(axes, scales, lws):
        surface_artists = []
        for s in surfaces0:
            ni, nj = s.shape[:2]

            if style == "scatter":
                if is_3d:
                    sc = ax.scatter(s[:, 0], s[:, 1], c = s[:, 2], cmap = cfg["cmap_scatter"], s=pt_size, linewidths=0)
                    surface_artists.append(("scatter", sc))

            elif style == "checkerboard":
                stepi  = int(max(ni // sc, 1))
                stepj  = int(max(nj // sc, 1))
                mask = _checkerboard_mask(s, stepi, stepj)
                pts  = s.reshape(-1, s.shape[-1])
                sc_a = ax.scatter(pts[ mask, 0], pts[ mask, 1],
                                  c=colors[0], s=pt_size, linewidths=0)
                sc_b = ax.scatter(pts[~mask, 0], pts[~mask, 1],
                                  c=colors[1], s=pt_size, linewidths=0)
                
                surface_artists.append(("checker", mask, sc_a, sc_b))

            elif style == "mesh":
                basei = int(max(ni//maxscales, 1))
                stepi  = basei*int(maxscales//sc)
                basej = int(max(nj//maxscales, 1))
                stepj  = basej*int(maxscales//sc)

                for i in list(range(0, ni, stepi)):
                    kill_long_edges(s[i], threshold = long_edge)
                for j in list(range(0, nj, stepj)):
                    kill_long_edges(s[:, j], threshold = long_edge)

                lines_i = [
                    ax.plot(s[i, :, 0], s[i, :, 1], color="black", lw=lw)[0]
                    for i in list(range(0, ni, stepi))
                ]
                lines_j = [
                    ax.plot(s[:, j, 0], s[:, j, 1], color="black", lw=lw)[0]
                    for j in list(range(0, nj, stepj)) 
                ]
                surface_artists.append(
                    ("mesh", ni, nj, stepi, stepj, lines_i, lines_j)
                )

        scale_artists.append(surface_artists)

    plt.tight_layout()

    def _iter_all_artists():
        yield from bg_artists
        for surf_list in scale_artists:
            for info in surf_list:
                if info[0] == "scatter":
                    yield info[1]
                elif info[0] == "checker":
                    yield info[2]; yield info[3]
                elif info[0] == "mesh":
                    yield from info[5]; yield from info[6]

    def _update(frame):
        t              = 2* (frame / max(frames - 1, 1))
        if t > 1:
            t = 2-t

        surfaces, full = _lerp(t)


        for bg in bg_artists:
            bg.set_offsets(full[:, :2])

        for surf_list in scale_artists:
            for s_idx, info in enumerate(surf_list):
                s   = surfaces[s_idx]
                pts = s.reshape(-1, s.shape[-1])

                if info[0] == "scatter":
                    _, sc = info
                    sc.set_offsets(pts[:, :2])

                elif info[0] == "checker":
                    _, mask, sc_a, sc_b = info
                    if pts.shape[1] == 3:
                        order       = np.argsort(pts[:, 2])
                        pts         = pts[order]
                        mask_sorted = mask[order]
                    else:
                        mask_sorted = mask
                    sc_a.set_offsets(pts[ mask_sorted, :2])
                    sc_b.set_offsets(pts[~mask_sorted, :2])

                elif info[0] == "mesh":
                    _, ni, nj, stepi, stepj, lines_i, lines_j = info
                    for k, i in enumerate(range(0, ni, stepi)):
                        lines_i[k].set_data(s[i, :, 0], s[i, :, 1])
                    for k, j in enumerate(range(0, nj, stepj)):
                        lines_j[k].set_data(s[:, j, 0], s[:, j, 1])

        return list(_iter_all_artists())

    ani = animation.FuncAnimation(
        fig, _update,
        frames=frames,
        interval=cfg["interval"],
        blit=True,
    )

    return fig, ani


# =========================================================
# PUBLIC API
# =========================================================
def sqplot(grid, verbose, style="checkerboard", animate=False,
           save=True, save_path="sqrnet/plot", cfg=None):
    """
    Render a structured grid as either a static figure or a morphing animation.

    The input grid is displayed using one of several rendering styles and may
    optionally be animated by continuously interpolating between the input grid
    and the corresponding identity grid.

    Parameters
    ----------
    grid : ndarray
        Structured grid of shape ``(..., D)``, where ``D`` is either 2 or 3.
    verbose : bool
        If ``True``, prints progress information while rendering or exporting.
    style : {"checkerboard", "mesh", "scatter"}, default="checkerboard"
        Rendering style.

        - ``"checkerboard"``: alternating colours on adjacent grid cells.
        - ``"mesh"``: draw the grid connectivity as a wireframe.
        - ``"scatter"``: draw only the grid points. In 3-D, points are coloured
        according to their depth after projection; in 2-D they are drawn in
        black.
    animate : bool, default=False
        If ``True``, generate an animation interpolating between the input grid
        and the identity grid. Otherwise, produce a single static figure.
    save : bool, default=True
        If ``True``, save the generated figure or animation to disk.
    save_path : str, default="sqrnet/plot"
        Output path used when ``save=True``.
    cfg : dict, optional
        Rendering configuration. Any omitted entries are filled with the default
        values returned by ``default_config()``. See
        ``help(default_config)`` for the complete list of supported options.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    ani : matplotlib.animation.FuncAnimation or None
        The animation object if ``animate=True``; otherwise ``None``.

    See Also
    --------
    default_config
        Default rendering options and their documentation.
    """
    grid = validate(grid)
    full_cfg = default_config()
    if cfg is not None:
        full_cfg.update(cfg)
    
    full_cfg.update({"style": style, 
                     "animate":animate, 
                     "save": save, 
                     "save_path": save_path
    })

    grid = atmost3D(grid, projection=full_cfg["projection"])

    animate   = full_cfg["animate"]
    save_path = full_cfg["save_path"]

    if animate:
        fig, ani = _build_animation(grid, full_cfg)
        if save_path is not None:
            has_ffmpeg = animation.writers.is_available("ffmpeg")
            suffix =  ".mp4" if has_ffmpeg else ".gif"
            writer = "ffmpeg" if has_ffmpeg else "pillow"
            save_path = _safe_path(save_path, suffix)
            if verbose >=1:
                print(f"figure will be saved at {save_path}")
                N = np.prod(grid.shape[:-1])
                rounded = 10 ** int(np.log10(N))
                if N >= 50_000 and suffix == ".gif":
                    if style == "scatter":
                        print(f"ffmpeg unavailable. Npoints = {rounded} and style is 'scatter' -> could take 1/2 minutes")   
                    else:
                        print(f"ffmpeg unavailable. Npoints = {rounded} -> could take 20/30 seconds")
            if has_ffmpeg:
                from matplotlib.animation import FFMpegWriter
                # On passe le fps ICI, lors de la création de l'objet
                writer = FFMpegWriter(fps=full_cfg["fps"], codec="h264", bitrate=-1)
                # Et on ne le passe PLUS dans .save()
                ani.save(save_path, writer=writer)
            else:
                writer = "pillow"
                # Avec le string "pillow", on doit passer le fps dans .save()
                ani.save(save_path, writer=writer, fps=full_cfg["fps"])
        else:
            if cfg["show"]:
                plt.show()
        return fig, ani

    else:
        static_cfg           = full_cfg.copy()
        static_cfg["frames"] = 1
        fig, _ = _build_animation(grid, static_cfg)
        if save_path is not None:
            save_path = _safe_path(save_path, ".png")
            print(f"figure will be saved at {save_path}")
            fig.savefig(save_path, bbox_inches="tight")
        if cfg["show"]:
                plt.show()
        return fig, None
