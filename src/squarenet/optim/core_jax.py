import numpy as np
import jax
import jax.numpy as jnp

from .jax.booster import loop_boost
from .jax.subgrid import random_subgrid_split
from .jax.hashtable import HashTable


# ============================================================================
# Public API
# ============================================================================

def jax_carthesian_sort(
    gridmap,
    points,
    method="fast",
    max_iter=100,
    verbose=2,
    loop=None,
    loopseq="decreasing",
):
    """
    Fully JIT-compatible Cartesian sorting.
    Multi-method dispatcher.
    """

    if verbose >= 2:
        print(f"jax working ({method}) ...")

    methods = {
        "fast": fast_carthesian_sort,
        "robust": robust_carthesian_sort,
        "ultimate": ultimate_carthesian_sort,
    }

    if method not in methods:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Expected one of {list(methods.keys())}"
        )

    return methods[method](
        gridmap,
        points,
        max_iter=max_iter,
        verbose=verbose,
        loop=loop,
        loopseq=loopseq,
    )


# ============================================================================
# Helpers
# ============================================================================

def _prepare(gridmap, points, loop, loopseq):

    g = jnp.asarray(gridmap, dtype=jnp.int32)
    points = jnp.asarray(points)

    gshape = np.asarray(g.shape)

    dims = np.where(gshape > 1)[0]

    if len(dims) == 0:
        raise ValueError("gridmap has no active dimensions")

    if loopseq == "decreasing":
        dims = dims[np.argsort(-gshape[dims])]

    elif loopseq == "random":
        dims = np.random.permutation(dims)

    else:
        raise ValueError(
            f"unknown loopseq {loopseq!r}, "
            f"should be 'decreasing' or 'random'"
        )

    dims = tuple(int(d) for d in dims)

    if loop is None:
        loop = loop_boost(points[:, dims])

    init_loop, circular_loop, end_loop = loop

    g = init_loop[g]

    return (
        g,
        dims,
        loop,
        circular_loop,
        end_loop,
    )


def _sort_pass(
    g,
    dims,
    circular_loop,
    it,
):

    disorder = jnp.array(0, dtype=jnp.int32)

    first_dim = dims[0]

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

    return g, disorder


def _build_fast_while(
    dims,
    circular_loop,
    max_iter,
):

    def body_fn(state):

        g = state["g"]
        it = state["it"]
        lc = state["learning_curve"]

        g, disorder = _sort_pass(
            g,
            dims,
            circular_loop,
            it,
        )

        lc = lc.at[it].set(disorder)

        return {
            "g": g,
            "disorder": disorder,
            "learning_curve": lc,
            "it": it + 1,
        }

    def cond_fn(state):

        return jnp.logical_and(
            state["disorder"] > 0,
            state["it"] < max_iter,
        )

    return cond_fn, body_fn


def _cleanup(
    g,
    points,
    end_loop,
    loop,
    loopseq,
    max_iter,
    verbose,
):

    g = end_loop[g]

    g, (lc, last_it) = fast_carthesian_sort(
        g,
        points,
        max_iter=max_iter,
        verbose=verbose,
        loop=loop,
        loopseq=loopseq,
    )

    return g, (lc, last_it)


# ============================================================================
# Fast Variant
# ============================================================================

def fast_carthesian_sort(
    gridmap,
    points,
    max_iter=100,
    verbose=2,
    loop=None,
    loopseq="decreasing",
):

    (
        g,
        dims,
        loop,
        circular_loop,
        end_loop,
    ) = _prepare(
        gridmap,
        points,
        loop,
        loopseq,
    )

    cond_fn, body_fn = _build_fast_while(
        dims=dims,
        circular_loop=circular_loop,
        max_iter=max_iter,
    )

    init_state = {
        "g": g,
        "disorder": jnp.array(1, dtype=jnp.int32),
        "learning_curve": jnp.zeros(
            max_iter + 1,
            dtype=jnp.int32,
        ),
        "it": jnp.array(0, dtype=jnp.int32),
    }

    final_state = jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_state,
    )

    sorted_grid = end_loop[final_state["g"]]

    learning_curve = final_state["learning_curve"]

    last_it = final_state["it"] - 1

    return sorted_grid, (
        learning_curve,
        last_it,
    )


# ============================================================================
# Robust Variant
# ============================================================================

def robust_carthesian_sort(
    gridmap,
    points,
    max_iter=100,
    verbose=2,
    loop=None,
    loopseq="decreasing",
):

    (
        g,
        dims,
        loop,
        circular_loop,
        end_loop,
    ) = _prepare(
        gridmap,
        points,
        loop,
        loopseq,
    )

    gshape = tuple(g.shape)

    def cond_fn(state):

        return jnp.logical_and(
            state["disorder"] > 0,
            state["it"] < max_iter,
        )

    def body_fn(state):

        g = state["g"]
        key = state["key"]
        it = state["it"]
        lc = state["learning_curve"]

        key, subkey = jax.random.split(key)

        subgrids = random_subgrid_split(
            subkey,
            gshape,
            dims,
        )

        disorder = jnp.array(0, dtype=jnp.int32)

        circular = False

        for d_id, (d, heuristic) in enumerate(
            zip(dims, circular_loop)
        ):

            g = jax.lax.cond(
                circular,
                lambda x: heuristic[x],
                lambda x: x,
                g,
            )

            circular = True

            for sub in subgrids[d_id]:

                disorder = disorder + jnp.sum(
                    jnp.diff(g[sub], axis=d) < 0,
                    dtype=jnp.int32,
                )

                g = g.at[sub].set(
                    jnp.sort(g[sub], axis=d)
                )

        lc = lc.at[it].set(disorder)

        return {
            "g": g,
            "key": key,
            "it": it + 1,
            "disorder": disorder,
            "learning_curve": lc,
        }

    init_state = {
        "g": g,
        "key": jax.random.PRNGKey(0),
        "it": jnp.array(0, dtype=jnp.int32),
        "disorder": jnp.array(1, dtype=jnp.int32),
        "learning_curve": jnp.zeros(
            max_iter + 1,
            dtype=jnp.int32,
        ),
    }

    state = jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_state,
    )



    g, (lc2, last_it2) = _cleanup(
        state["g"],
        points,
        end_loop,
        loop,
        loopseq,
        max_iter,
        verbose,
    )

    lc1 = state["learning_curve"]
    last_it1 = state["it"]

    return g, (
        jnp.concatenate([lc1, lc2]),
        last_it1 + last_it2,
    )


# ============================================================================
# Ultimate Variant
# ============================================================================

def ultimate_carthesian_sort(
    gridmap,
    points,
    max_iter=100,
    verbose=2,
    loop=None,
    loopseq="decreasing",
):

    (
        g,
        dims,
        loop,
        circular_loop,
        end_loop,
    ) = _prepare(
        gridmap,
        points,
        loop,
        loopseq,
    )

    # ------------------------------------------------------------------
    # Phase 1 : robust preconditioning
    # ------------------------------------------------------------------

    g, (lc1, last_it1) = robust_carthesian_sort(
        g,
        points,
        max_iter=max_iter,
        verbose=verbose,
        loop=loop,
        loopseq=loopseq,
    )

    # ------------------------------------------------------------------
    # Phase 2 : HashTable refinement
    # ------------------------------------------------------------------

    htable = HashTable(g, dims)

    circular = False

    for _ in range(4 * max_iter):

        for heuristic in circular_loop:

            if circular:
                htable.gtable = heuristic[htable.gtable]

            circular = True

            htable.sort()

    # ------------------------------------------------------------------
    # Final cleanup
    # ------------------------------------------------------------------

    g, (lc2, last_it2) = _cleanup(
        htable.gtable,
        points,
        end_loop,
        loop,
        loopseq,
        max_iter,
        verbose,
    )

    return g, (
        jnp.concatenate([lc1, lc2]),
        last_it1 + last_it2,
    )

