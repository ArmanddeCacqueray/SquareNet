import numpy as np
from warnings import warn

from .core import carthesian_sort
from .artist import sqplot, default_config
from .sampler import samplepoints
from .neighbormap import neighbormap
from .utils import dualgrid, dualgridflat, to_backend, printmatrix


class SquareNet:
    """
    Grid straightening algorithm for D-dimensional point clouds.

    SquareNet reorganizes an unordered set of points into a structured grid
    by iteratively sorting indices along each spatial axis. The goal is to
    enforce a topology where neighboring points in the original space
    are also close on the grid.

    The grid is internally represented as an array of indices of shape
    ``(N1, ..., ND)``, where D is the dimensionality of the embedding space.

    Parameters
    ----------
    gridshape : tuple of int
        Shape of the target grid. The product of these dimensions must 
        exactly match the total number of input points (bijective gridification).
    max_iter : int, default=100
        Maximum number of iterations allowed for the sorting procedure.
    warnings_ : bool, default=True
        If True, emits a ``ConvergenceWarning`` if the grid is not 
        perfectly ordered at the end of the process.
    backend : {"numpy", "torch" or "jax"}, default="numpy"
        input and output backend.

    Attributes
    ----------
    grid : ndarray
        The structured grid of indices, with shape ``gridshape``.
    invert_grid : ndarray, jax or numpy
        Inverse mapping from point indices to grid coordinates.
    invgridflat : ndarray, jax or numpy
        Flattened version of the inverse mapping for fast indexing.
    learning_curve : list of float
        History of the disorder metric across iterations.
    
    1. Run ``fit()`` with your preferred backend.
    2. Call ``SquareNet.map()`` on any feature indexed like the points (N, *C)
    To build a tensor version of it (*G, *C) based on the points learned multi-indexes
    """

    def __init__(self, gridshape, max_iter = 500, warnings_=True, 
                 verbose = 2, backend = "numpy", device = "cpu"):
        self.gridshape = tuple(gridshape)
        self.D = len(self.gridshape)
        self.N = int(np.prod(self.gridshape))
        self.max_iter = max_iter
        self.verbose = verbose
        self.backend = backend
        self.device = device

        if backend not in {"numpy", "jax", "torch"}:
            raise ValueError(f"Unknown backend '{backend}' Should be 'numpy' 'torch' or 'jax'")
        
        if backend == "numpy":
            self.xp = np

        elif backend == "jax":
            import jax.numpy as jnp
            self.xp = jnp

        elif backend == "torch":
            import torch
            self.xp = torch

        self.warnings_ = warnings_

        self.plot_config = default_config()

        self._reset_state()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def to_backend(self, x):
        return to_backend(x, backend = self.backend, device = self.device, warnings_ = self.warnings_)
    
    def _reset_state(self):
        """Reset internal state to initial configuration."""
        if self.backend == "torch":
           self.grid = self.xp.arange(
                self.N,
                dtype=self.xp.int32,
                device=self.device,
            ).reshape(self.gridshape)

        else:
            self.grid = self.xp.arange(
                self.N,
                dtype=self.xp.int32,
            ).reshape(self.gridshape)
        self.invert_grid = dualgrid(self.grid, self.xp, self.N, self.gridshape, self.D)
        self.invgridflat = dualgridflat(self.grid, self.xp, self.N)

        self.points = None
        self.pointsmaped = None

        self.learning_curve = []
        self.fitted = False

    def _validate_points(self, points):
        """Validate input point cloud."""
        if isinstance(points, str):
            points = samplepoints(method=points, size=(self.N, self.D))

        points = self.to_backend(points)
        N, D = points.shape

        assert N == self.N, (
            f"Input points ({N}) must match grid size ({self.N})."
            "For injective gridification, explicitly add fictive points with +- infinite coordinates",
            "Which will then be placed in the corners of the grid"
        )
        assert D == self.D, (
            f"Input dimension ({D}) must match D={self.D}. "
            "For manifold data, explicitly include singleton dimensions, e.g. gridshape = (100, 100, 1)."
        )

        return points

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, points, method = "fast"):
        """
        Fit the grid to a point cloud.

        Parameters
        ----------
        points : ndarray of shape (N, D) or str
            Input point cloud or sampling method name.
        
        method : fast or robust.
            robust can be up to 5 time slower, but
            will probably give a better grid.

        Returns 
        -------
        None
            The method updates the internal state of the grid in-place.
        
        see ``squarenet.core_numpy``

        Note
        -------
        - robust fit method is 100% Numpy, which means there will
            be some overhead backend -> numpy -> backend during the fit

        - fit is almost jitable for jax, but not as is.
        Fot a jax jit friendly fit, see ``squarenet.squarenet.jitable_fit``
        """
        if method == "robust":
            if self.fitted:
                self._reset_state()

        max_iter = self.max_iter
        verbose = self.verbose
        def _log(msg):
            if verbose >= 2:
                print(msg)

        def _section(title):
            if verbose >= 2:
                print(f"\n=== {title} ===")

        points = self._validate_points(points)
        self.points = points

        _log("Starting gridification... available method [fast, robust]")
        _log(f"selected {method}")

        _section(f"Carthesian sort")

        self.grid, lc = carthesian_sort(self.grid, points, max_iter=max_iter, method = method,
                                        backend = self.backend, verbose = self.verbose)

        # --------------------------------------------------
        # Finalization
        # --------------------------------------------------
        if isinstance(lc, tuple):
            lc, last_iter = lc
            self.learning_curve = list(lc)[:last_iter +1]
        else:
            self.learning_curve = list(lc)

        last_error = self.learning_curve[-1]
        last_iter = len(self.learning_curve) - 1

        _section("Final Status")

        if verbose >= 1:
            if last_error == 0:
                print(f"succesfully sorted at iteration {last_iter}")
            else:
                print(f"not fully sorted (iter = {max_iter}, error={last_error/self.N:.4%})")

                if self.warnings_:
                    warn(
                        "Sorting did not converge to zero. "
                        "Consider increasing `max_iter`.",
                        ConvergenceWarning,
                        stacklevel=2,
                    )

        # --------------------------------------------------
        # Update mappings
        # --------------------------------------------------
        self.invert_grid = dualgrid(self.grid, self.xp, self.N, self.gridshape, self.D)
        self.invgridflat = dualgridflat(self.grid, self.xp, self.N)
        self.fitted = True

        self.pointsmaped = self.map(self.points)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------
    def map(self, features):
        """
        Map cloud data to grid structure.

        Parameters
        ----------
        features : jax, or numpy ndarray of shape (N, *C)

        Returns
        -------
        jax or numpy ndarray of shape (*gridshape, *C)

        """
        return features[self.grid]

    def invert_map(self, features):
        """
        Map grid data back to cloud ordering.

        Parameters
        ----------
        features : jax or numpy ndarray of shape (*gridshape, *C)

        Returns
        -------
        jax or numpy ndarray of shape (N, *C)
        """
        return features.reshape( 
            self.N, *features.shape[self.D:]
        )[self.invgridflat] 

    def mapidx(self, index):
        """
        Cloud index -> grid index.

        Returns
        -------
        tuple of arrays
            directly usable for indexing.
        """

        coords = self.invert_grid[index]

        return tuple(coords.T)

    def invert_mapidx(self, index):
        """
        Grid index -> cloud index.
        """

        index = self.xp.asarray(index)
        if index.ndim == 1:
            return self.grid[tuple(index)]

        return self.grid[tuple(index.T)]
    
    def search_sorted(self, X, n_iter=2, side = "left"):
        """
        X must be a SINGLE point, search_sorted will return a multiindex IX in the grid.
        IX is the grid index of PROBABLY one of the closest points to X in the first (side = 'left') 
        or last (side = 'right') quadrant (relative to X) of the space. 
        The search is a greedy coordinate descent: each dimension is refined by applying 1D searchsorted 
        on row, then columns... Augmenting n_iter increases accuracy.
        """
        IX = self.xp.zeros(self.D, dtype=self.xp.int32)   
        Y =  self.pointsmaped

        gdims = self.gridshape

        side_offset = -1 if side == "left" else 0

        for _ in range(n_iter):
            for d in range(self.D):
                IXn = list(IX) + [d] 
                IXn[d] = slice(None)
                newI = self.xp.searchsorted(Y[tuple(IXn)], X[d], side = side) + side_offset  
                if newI < 0:
                    newI = 0
                if newI > gdims[d] - 1:
                    newI = gdims[d] - 1
                IX[d] = newI
        return IX

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def plot(self,
        style="checkerboard",
        animate=False,
        save = True,
        save_path="sqrnet/plot",
        **kwargs
    ):
        """
        Display the mapped grid as a static figure or animation.

        Many rendering options (scales, colors, figsize, DS, frames …)
        can be passed as keyword arguments or configured once via
        ``self.plot_config[key] = value``.
        print(self.plot_config) to see all available parameters).

        Parameters
        ----------
        style : str
            Rendering style: ``"checkerboard"``, ``"mesh"`` or ``"scatter"``.
        animate : bool
            If True, produce a morphing animation (grid → identity).
            If False, render a static plot.
        save_path : str or None
            None  → display with ``plt.show()``.
            str   → save to disk

        Returns
        -------
        fig : matplotlib.figure.Figure
        ani : matplotlib.animation.FuncAnimation or None

        see ``squarenet.artist``
        """
        gridpoints = self.pointsmaped
        if hasattr(gridpoints, "detach"):
            gridpoints = gridpoints.detach().cpu().numpy()
        gridpoints = np.ascontiguousarray(np.asarray(gridpoints))
        if save == False:
            save_path = None
        plot_config = self.plot_config.copy()
        plot_config.update(kwargs)
        plot_config.update({"style": style, "animate": animate, "save_path": save_path})

        return sqplot(gridpoints, self.verbose,
                      style = style, animate = animate, 
                      save = save, save_path = save_path,
                      cfg = plot_config)
    
    def neighbormap(self, max_sample_size = 1_000_000, max_window_size = 41, criterion="rank", thresholdcut=1,
                projection=(0, 1), log2=False):
        """
        Compute and display a neighborhood map from gridded points.

        For each point X (or a subsample of sample_size if the dataset is too large),
        scans a square window of radius wr = ws//2 in the grid centered on X, and adds 1
        to each cell if the point Y found in the cell matches the criterion.

        Parameters
        ----------
        max_sample_size: int, default 1 million
            sample size for the number of pairs (X, Y)
        max_window_size: int, default=21
            Note that ``wr``, the window radius, is defined as
            ``wr = window_size // 2``.      
            Search window size, i.e. the size of the window in which the
            neighbor map is computed, such that:      
            ``offset(gridindex(X), gridindex(Y)) <= wr``    
            where ``offset`` denotes the L∞ norm on grid indices.
        
            Warning
            -------
            You will get no information, and thus no garantee at all, 
            on what happens outside the search window. Conversely, using 
            a very large search window dilutes the information, since the 
            probability of sampling pairs for a given grid offset decreases.
        
            Choosing the window size is therefore a tradeoff.
            
        criterion : str, optional
            Method used to define neighborhoods ("rank" or "value").
        thresholdcut : int or float, optional Cutoff threshold for connections, interpreted according to the criterion.
            - If criterion is "rank": thresholdcut = k means Y matches X
            if Y is among the k nearest points to X.
            - If criterion is "value": thresholdcut is a distance threshold,
            and Y matches X if the distance(X, Y) <= thresholdcut.
        projection : tuple of int, optional
            Axes of the grid used for 2D projection.
        log2 : bool, optional
            If True, applies log2 scaling to counts.

        Returns
        -------
        None
            Prints the resulting neighborhood matrix.
        
        see ``squarenet.neighbormap``
        """

        nbmap = neighbormap(
            self.pointsmaped,
            self.mapidx,
            sample_size = max_sample_size,
            windowradius = max_window_size//2
            criterion=criterion,
            thresholdcut=thresholdcut,
            projection=projection
        )

        if log2:
            arr = np.log2(nbmap + 0.01).astype(int)
            print(f"neighbor map on axes {projection} (log2 count)")
        else:
            arr = nbmap.astype(int)
            print(f"neighbor map on axes {projection}")
        
        arr[nbmap == 0] = -1 #invalid log for 0
        
        printmatrix(arr)

def jitable_fit(points, grid, max_iter):
    """
    Pure JAX version of fit.
    Jitable via jax.jit(jitable_fit).

    Typical usecase:
    max_iter = 500
    @jax.jit
    def jited_fit(points, grid):
        return jitable_fit(points, grid, max_iter)

    I = 100
    N = I*I
    points = samplepoints("holy", size = (N, 2))
    sn = SquareNet(gridshape = (I, I), backend = "jax")
    grid = sn.grid

    grid, learning_curve, last_iter = jited_fit(points, grid)
    updatenet(sn, points, grid, learning_curve, last_iter)
    
    Returns
    -------
    grid, learning_curve, last_iter
    """
    grid, lc_it = carthesian_sort(
        grid, points,
        max_iter=max_iter,
        method="fast",
        backend="jax",
        verbose=0,
    )
    learning_curve, last_iter = lc_it
    return grid, learning_curve, last_iter

def updatenet(sqnet, points, grid, learning_curve, last_iter):
    sqnet.points = points
    sqnet.grid = grid
    sqnet.invert_grid = dualgrid(sqnet.grid, sqnet.xp, sqnet.N, sqnet.gridshape, sqnet.D)
    sqnet.invgridflat = dualgridflat(sqnet.grid, sqnet.xp, sqnet.N)
    sqnet.learning_curve = list(learning_curve)[:last_iter +1]
    sqnet.fitted = True
    sqnet.pointsmaped = sqnet.map(points)

class ConvergenceWarning(UserWarning):
    """Raised when optimization does not converge."""
    pass
