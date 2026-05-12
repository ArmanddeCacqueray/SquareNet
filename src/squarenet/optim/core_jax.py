"""
Jax implementation of carthesian sort
Fully JIT-compatible.

Typical usage
-------------
>>> @jax.jit
... def step(g, pts_flat):
...     g, lc = jax_carthesian_sort(g, pts_flat)
        pts_grid = pts_flat[g] #(*G, D)
        #apply some logic on the gridified points...
        #... then back to flat structure
        pts_flat = pts_grid.reshape(-1, D) #(N, D)
...     return g, pts_flat
"""

import numpy as np
import jax
import jax.numpy as jnp


# ============================================================================
# Helpers
# ============================================================================

def _active_dims(g):
    """
    Return active dimensions sorted by descending grid size.

    Dimensions of size 1 are ignored.
    """
    gshape = np.asarray(g.shape)

    dims = np.where(gshape > 1)[0]

    # largest dimensions first
    dims = dims[np.argsort(-gshape[dims])]

    return tuple(int(d) for d in dims)


# ============================================================================
# Integer / permutation boosts
# ============================================================================

def integer_boost(points):
    """
    Build rank permutations for each coordinate dimension.

    Parameters
    ----------
    points : (N, D) array

    Returns
    -------
    ranks : list[(N,) int32]
        ranks[d][i] gives rank of point i along dimension d.
    """
    points = jnp.asarray(points)

    N, D = points.shape

    ranks = []

    for d in range(D):

        order = jnp.argsort(points[:, d])

        rank = (
            jnp.zeros(N, dtype=jnp.int32)
            .at[order]
            .set(jnp.arange(N, dtype=jnp.int32))
        )

        ranks.append(rank)

    return tuple(ranks)


def loop_boost(points):
    """
    Build circular permutation heuristics.

    Returns
    -------
    init_loop
    circular_loop
    end_loop
    """
    int_boost = integer_boost(points)

    N = int_boost[0].shape[0]

    identity = jnp.arange(N, dtype=jnp.int32)

    h_sources = (identity,) + int_boost
    h_targets = int_boost + (identity,)

    loop = tuple(
        jnp.zeros(N, dtype=jnp.int32).at[h].set(hp)
        for h, hp in zip(h_sources, h_targets)
    )

    init_loop = loop[0]
    end_loop = loop[-1]

    circular_loop = list(loop[:-1])

    # ensure proper round-trip on first pass
    circular_loop[0] = init_loop[end_loop]

    return (
        init_loop,
        tuple(circular_loop),
        end_loop,
    )


# ============================================================================
# Main while-loop builder
# ============================================================================

def _build_while_body(
    dims,
    circular_loop,
    max_iter,
):
    """
    Build JAX while_loop functions.

    IMPORTANT:
    ----------
    dims MUST remain Python static integers.

    Dynamic axes inside jnp.sort/jnp.diff are illegal under JIT.
    """

    first_dim = dims[0]

    def body_fn(state):

        g = state["g"]
        disorder = jnp.array(0, dtype=jnp.int32)

        learning_curve = state["learning_curve"]
        it = state["it"]

        # ------------------------------------------------------------
        # Python-unrolled loop
        # axes remain STATIC during tracing
        # ------------------------------------------------------------

        for k, d in enumerate(dims):

            heuristic = circular_loop[k]

            skip = jnp.logical_and(
                it == 0,
                d == first_dim,
            )

            g = jax.lax.cond(
                skip,
                lambda x: x,
                lambda x: heuristic[x],
                g,
            )

            disorder = disorder + jnp.sum(
                jnp.diff(g, axis=d) < 0,
                dtype=jnp.int32,
            )

            g = jnp.sort(g, axis=d)

        learning_curve = learning_curve.at[it].set(disorder)

        return {
            "g": g,
            "disorder": disorder,
            "learning_curve": learning_curve,
            "it": it + 1,
        }

    def cond_fn(state):

        return jnp.logical_and(
            state["disorder"] > 0,
            state["it"] < max_iter,
        )

    return cond_fn, body_fn


# ============================================================================
# Public API
# ============================================================================

def jax_carthesian_sort(
    gridmap,
    points,
    max_iter=100,
    loop=None,
):
    """
    Fully JIT-compatible Cartesian sorting.

    Parameters
    ----------
    gridmap : ndarray[int]
        Grid of point indices.

    points : (N, D) ndarray
        Point cloud.

    max_iter : int
        Maximum optimization iterations.

    loop : optional precomputed loop structure
        Output of loop_boost(...)

    Returns
    -------
    sorted_grid : ndarray[int32]
    learning_curve : ndarray[int32]
    """

    # ---------------------------------------------------------------------
    # Input normalization
    # ---------------------------------------------------------------------

    g = jnp.asarray(gridmap, dtype=jnp.int32)

    points = jnp.asarray(points)

    # ---------------------------------------------------------------------
    # Static active dimensions
    # ---------------------------------------------------------------------

    dims = _active_dims(g)

    if len(dims) == 0:
        raise ValueError("gridmap has no active dimensions")

    # ---------------------------------------------------------------------
    # Build permutations
    # ---------------------------------------------------------------------

    if loop is None:

        loop = loop_boost(points[:, dims])

    init_loop, circular_loop, end_loop = loop

    # ---------------------------------------------------------------------
    # Initial permutation
    # ---------------------------------------------------------------------

    g = init_loop[g]

    # ---------------------------------------------------------------------
    # Build while-loop
    # ---------------------------------------------------------------------

    cond_fn, body_fn = _build_while_body(
        dims=dims,
        circular_loop=circular_loop,
        max_iter=max_iter,
    )

    # ---------------------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------------------

    init_state = {
        "g": g,
        "disorder": jnp.array(1, dtype=jnp.int32),
        "learning_curve": jnp.zeros(
            max_iter + 1,
            dtype=jnp.int32,
        ),
        "it": jnp.array(0, dtype=jnp.int32),
    }

    # ---------------------------------------------------------------------
    # Main optimization loop
    # ---------------------------------------------------------------------

    final_state = jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_state,
    )

    # ---------------------------------------------------------------------
    # Undo initial permutation
    # ---------------------------------------------------------------------

    sorted_grid = end_loop[final_state["g"]]

    last_it = final_state["it"] - 1

    learning_curve = final_state["learning_curve"]

    return sorted_grid, (learning_curve, last_it)