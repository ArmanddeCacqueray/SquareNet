from importlib import import_module

_BACKENDS = {
    "numpy": ".optim.core_numpy",
    "torch": ".optim.core_torch",
}

def _load_backend(name: str):
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{name}', expected one of {list(_BACKENDS)}"
        )
    module = import_module(_BACKENDS[name], package=__package__)
    return getattr(module, f"{name}_cartesian_sort")


def cartesian_sort(
    gridmap,
    points,
    max_iter=100,
    method="fast",
    backend="numpy",
    loop=None,
    loopseq="decreasing",
    verbose=2,
):
    """
    Supported:
    ----
    method [fast, robust, ultimate]
    backend [numpy, torch]


    Goal
    ----
    Given an unordered point cloud:

        points.shape = (N, D)

    rearrange point indices into a structured cartesian grid:

        grid.shape = (N1, N2, ..., ND)

    such that neighbouring cells of the grid contain spatially
    coherent points.

    ============================================================
    GRID INTERPRETATION
    ============================================================

    A grid index map bijectively to a point index:

        point index: n <=> grid index: f(n) = (n1, n2, ..., nD)

    Each grid axis defines a local neighbour relation.

    For axis d, let define:

        next(n, d) = f-1(n1,..., nd+1,...,nD)

    such that P(next(n,d)) is the neighbour obtained by 
    incrementing the d-th grid coordinate.

    The objective is therefore:

    nearby euclidean points -> nearby grid cells

    Note that the reciprocal property

        nearby grid cells -> nearby euclidean points

    is NOT guaranteed.

    Indeed, datasets may contain cracks, holes, disconnected
    clusters, folds, or other topological singularities.

    Gridification will naturally tend to close cracks, stitch
    nearby boundaries together and overlap the folds.

    This algorithm is primarily designed for speed and scalability
    on large datasets. It is NOT an optimal transport solver.

    Therefore, users should always inspect the resulting grid,
    especially near boundaries where geometric distortions are
    more likely to appear.


    ============================================================
    HEURISTIC ORDERING PRINCIPLE
    ============================================================

    For each spatial dimension d, define d spatial heuristics:

        H_d(point) -> scalar
    
    Heuristics must be orthogonal and monotonic
    Typical choice: cartesian heuristics

        H_0 = x
        H_1 = y
        H_2 = z
        ...
    One could think on an improved version of the algorithm
    which would learn heuristics that best fit the dataset
    but cartesian are already pretty good

    The desired property is:

        H_d(P(n)) <= H_d(P(next(n, d)))

    for every point index n and every axis d.

    In words:

        values of heuristic d should increase
        along grid axis d.


    ============================================================
    DISORDER METRIC
    ============================================================

    A local inversion occurs when:

        H_d(P(n)) > H_d(P(next(n, d)))

    The total disorder is simply the number of such violations.

    Pseudo-definition:

        disorder =
            sum over all axes d
            sum over all point pairs (current, next)
            of inversion count


    ============================================================
    FAST METHOD
    ============================================================

    The simplest strategy consists in repeatedly sorting each
    grid axis independently.

    Pseudo-code
    ------------

    initialize gridmap

    for iteration in range(max_iter):

        for axis d in range(D):

            sort grid indexes along axis d
            using heuristic H_d

        compute disorder

        if disorder == 0:
            stop


    Properties
    ----------

    Advantages:
        - extremely fast
        - fully vectorized
        - memory efficient
        - surprisingly effective

    Limitations:
        - for loop on the axes
         -> early axes dominate later ones
         -> result in weak axes
        - only adjacent comparison is a weak accuracy criterion
         -> disorder is blind on what happens on diagonals
         -> may converge toward local minima


    ============================================================
    ROBUST METHOD
    ============================================================

    To reduce axis domination and improve stability,
    sorting is performed only on random independent subgrids at 
    each step, thus learning is much more progressiv

    At every iteration:

        - each axis is randomly partitioned
        - only selected cartesian sub-blocks are sorted
        - different blocks are used for different axes

    Result:

        - smoother convergence
        - better axis symmetry
        - fewer local minima
        - improved isotropy


    Pseudo-code
    ------------

    for iteration:

        generate random cartesian subgrids

        for axis d:

            for subgrid in selected_subgrids[d]:

                sort only inside this subgrid

        compute disorder


    ============================================================
    ULTIMATE METHOD
    ============================================================

    Even robust cartesian sorting still treats grid lines
    as parallel and therefore almost independent.

    This can produce:

        - stratification
        - layered artifacts
        - weak coupling between parallel slices

    To solve this, a second refinement stage is introduced.

    The grid is embedded into a "hash table"
    containing shifted/sheared copies of the grid.

    Repeated cyclic shears produce strong cross-line coupling.

    This allows information to propagate between previously
    independent cartesian lines.


    High-level idea
    ----------------

    repeat:

        shear grid into staggered hash table

        sort along one axis

        project back to grid



    Effect
    ------

    The repeated shearing progressively destroys artificial
    parallel structures and greatly reduces stratification.
    ============================================================
    SUMMARY
    ============================================================

    FAST:
        deterministic global line sorting (cartesian sort)

    ROBUST:
        stochastic partial sorting
        preserving axis symmetry

    ULTIMATE:
        sheared multi-pass relaxation
        reducing stratification artifacts


    ============================================================
    COMPLEXITY
    ============================================================

    A few hundreds iterations will be largely 
    enough for convergence in most of the cases.

    Let:

        N = total number of points

    Each iteration performs approximately:

        O(N (log N +D)) operations

    with highly vectorized NumPy operations.
    """

    kwargs = {
        "method": method,
        "max_iter": max_iter,
        "loop": loop,
        "loopseq": loopseq,
        "verbose": verbose,
    }

    fn = _load_backend(backend)
    return fn(gridmap, points, **kwargs)