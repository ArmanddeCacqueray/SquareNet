def carthesian_sort(
    gridmap,
    points,
    max_iter=100,
    method = "fast",
    backend="numpy",
    loop=None,
    verbose=2,
):
    """"
    =============================================
    =============================================
     FAST METHOD
    =============================================
    =============================================
    PSEUDO-CODE: convert unstructured point cloud (N, D) 
    to a grid (N1, ..., ND, D) by iteratively sorting 
    the d-est heuristic along the d-est axis

    We can see the grid as D iterators on the points 
    such that axis-d iterator maps  the point 
    P(n ~ n1...nd...nD) to the "next" point
    Pnext(n ~ n1...nd+1...nD). Let call it Pnext(n, d)

    We can select D euclidian heuristics
    H(d): (x, y, z,...) -> value which we want to 
    be increasing along the d-est axis of the grid
    It turns out that H(0) = x, H(1) = y,.... is
    already a pretty good heuristic.

    So the goal is simply to ensure that all
    points are sorted along the grid, in the
    sense that for all point P(n) and axis d,
    H(d)(P(n)) <= H(d)(Pnext(n, d))

    We can compute a grid disorder parameter which is
    just the counts of all P, Pnext which breaks this 
    inequality
    =============================================
    Sort_increasing is then pretty simple:
    For learning step in (1, Max_iter = 100)
        For d in (1, D):
            sort heuristic d along axis d.
        Check disorder
        If disorder = 0, we are done !
    =============================================
    =============================================
    ROBUST METHOD:
    same but at each step only a subgrid of the
    grid is sorted to make learning progressive
    and avoid breaking symetry on the dims
    =============================================
    =============================================
    ULTIMATE METHOD:
    same but at each step neigbor lines are 
    tangled and sorted together  to avoid
    stratification.
    =============================================
    =============================================
    """
    if method == "robust":
        from .optim.core_robust import robust_carthesian_sort
        return robust_carthesian_sort(
            gridmap,
            points,
            max_iter=max_iter,
            verbose=verbose,
            loop=loop,
            backend = backend,
        )
    elif method == "ultimate":
        from .optim.core_robust import robust_carthesian_sort
        from .optim.core_ultimate import tangled_carthesian_sort
        gridmap, _ = robust_carthesian_sort(
            gridmap,
            points,
            max_iter=max_iter,
            verbose=verbose,
            loop=loop,
            backend = backend,
        )
        return tangled_carthesian_sort(
            gridmap, 
            points, 
            max_iter=max_iter, 
            verbose=verbose, 
            loop=loop,
            backend = backend,
        )
    else:
        if method != "fast":
            raise ValueError(
                f"Unknown method '{method}', expected 'fast', 'robust' or 'ultimate'"
            )    
    if backend == "numpy":
        from .optim.core_numpy import np_carthesian_sort
        return np_carthesian_sort(
            gridmap,
            points,
            max_iter=max_iter,
            verbose=verbose,
            loop=loop,
        )

    elif backend == "jax":
        from .optim.core_jax import jax_carthesian_sort

        if verbose >= 2:
            print("jax working ...")

        return jax_carthesian_sort(
            gridmap,
            points,
            max_iter=max_iter,
            loop=loop,
        )

    elif backend == "torch":
        from .optim.core_torch import torch_cartesian_sort

        if verbose >= 2:
            print("torch working ...")

        return torch_cartesian_sort(
            gridmap,
            points,
            max_iter=max_iter,
            loop=loop,
        )

    else:
        raise ValueError(
            f"Unknown backend '{backend}', expected 'numpy', 'jax' or 'torch'"
        )
