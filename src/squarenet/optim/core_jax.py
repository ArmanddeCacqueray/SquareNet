import numpy as np
import jax
import jax.numpy as jnp

from .jax.booster import loop_boost
from .jax.subgrid import random_subgrid_split
from .jax.hashtable import HashTable

# ============================================================================
# Public API
# ============================================================================

def jax_carthesian_sort(gridmap, points, method="fast", max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    """Fully JIT-compatible Cartesian sorting (Multi-method Dispatcher)."""
    if verbose >= 2:
        print(f"jax working ({method}) ...")

    methods = {
        "fast": fast_carthesian_sort,
        "robust": robust_carthesian_sort,
        "ultimate": ultimate_carthesian_sort,
    }

    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Expected one of {list(methods.keys())}")

    return methods[method](
        gridmap, points, max_iter=max_iter, verbose=verbose, loop=loop, loopseq=loopseq
    )

# ============================================================================
# Helpers
# ============================================================================

def _prepare(gridmap, points, loop, loopseq):
    g = jnp.asarray(gridmap, dtype=jnp.int32)
    points = jnp.asarray(points)
    gshape = np.asarray(g.shape)
    
    dims = np.where(gshape > 1)[0]

    if loopseq == "decreasing":
        dims = dims[np.argsort(-gshape[dims])]
    elif loopseq == "random":
        dims = np.random.permutation(dims)
    else:
        raise ValueError(f"unknown loopseq {loopseq!r}, should be 'decreasing' or 'random'")

    dims = tuple(int(d) for d in dims)

    if loop is None:
        loop = loop_boost(points[:, dims])
    init_loop, circular_loop, end_loop = loop

    g = init_loop[g]
    return g, dims, loop, circular_loop, end_loop


def _check_convergence(g, dims, circular_loop):
    # Initialize disorder as a JAX array, not a Python integer
    disorder = jnp.array(0, dtype=jnp.int32)
    
    for k, d in enumerate(dims):
        g = circular_loop[k][g]
        disorder = disorder + jnp.sum(jnp.diff(g, axis=d) < 0, dtype=jnp.int32)
        
    return disorder


def _cleanup(g, points, end_loop, loop, loopseq, max_iter, verbose):
    g = end_loop[g]
    g, (lc, last_it) = fast_carthesian_sort(
        g, points,
        max_iter=max_iter, verbose=verbose,
        loop=loop, loopseq=loopseq,
    )
    return g, (lc, last_it)


# ============================================================================
# Fast Variant
# ============================================================================

def _build_while_body_chunked(dims, circular_loop, max_iter, chunk_size=10):
    def body_fn(state):
        g = state["g"]
        learning_curve = state["learning_curve"]
        big_it = state["big_it"]

        # 10 blind unrolled iterations without any branches/conds
        for _ in range(chunk_size):
            for k, d in enumerate(dims):
                g = circular_loop[k][g]
                g = jnp.sort(g, axis=d)

        # Single convergence check at the end of the big step
        disorder = _check_convergence(g, dims, circular_loop)

        learning_curve = learning_curve.at[big_it].set(disorder)

        return {
            "g": g,
            "disorder": disorder,
            "learning_curve": learning_curve,
            "big_it": big_it + 1,
        }

    def cond_fn(state):
        return jnp.logical_and(
            state["disorder"] > 0,
            state["big_it"] < (max_iter // chunk_size),
        )

    return cond_fn, body_fn


def fast_carthesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    g, dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)
    
    chunk_size = 10
    num_big_steps = max_iter // chunk_size

    # Handled first dim of it=0 manually to bypass the initial skip condition
    first_dim = dims[0]
    for k, d in enumerate(dims):
        if d != first_dim:
            g = circular_loop[k][g]
        g = jnp.sort(g, axis=d)

    cond_fn, body_fn = _build_while_body_chunked(
        dims=dims,
        circular_loop=circular_loop,
        max_iter=max_iter,
        chunk_size=chunk_size
    )

    init_state = {
        "g": g,
        "disorder": jnp.array(1, dtype=jnp.int32),
        "learning_curve": jnp.zeros(num_big_steps, dtype=jnp.int32),
        "big_it": jnp.array(0, dtype=jnp.int32),
    }

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    
    sorted_grid = end_loop[final_state["g"]]
    last_it = final_state["big_it"] * chunk_size
    learning_curve = final_state["learning_curve"]

    return sorted_grid, (learning_curve, last_it)


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

    g,dims,loop,circular_loop,end_loop=_prepare(
        gridmap,
        points,
        loop,
        loopseq,
    )

    gshape=tuple(g.shape)

    chunk_size=10
    n_big_steps=max_iter//chunk_size

    def cond_fn(state):
        return (
            (state["disorder"]>0)
            & (state["big_it"]<n_big_steps)
        )

    def body_fn(state):

        g=state["g"]
        key=state["key"]
        big_it=state["big_it"]
        lc=state["learning_curve"]

        def chunk_step(_,carry):

            g,key,circular=carry

            key,subkey=jax.random.split(key)

            subgrids=random_subgrid_split(
                subkey,
                gshape,
                dims,
            )

            for d_id,(d,heuristic) in enumerate(
                zip(dims,circular_loop)
            ):

                g=jax.lax.cond(
                    circular,
                    lambda x:heuristic[x],
                    lambda x:x,
                    g,
                )

                circular=True

                for sub in subgrids[d_id]:

                    g=g.at[sub].set(
                        jnp.sort(g[sub],axis=d)
                    )

            return g,key,circular

        g,key,_=jax.lax.fori_loop(
            0,
            chunk_size,
            chunk_step,
            (
                g,
                key,
                False,
            ),
        )

        disorder=_check_convergence(
            g,
            dims,
            circular_loop,
        )

        lc=lc.at[big_it].set(disorder)

        return {
            "g":g,
            "key":key,
            "big_it":big_it+1,
            "disorder":disorder,
            "learning_curve":lc,
        }

    init_state={
        "g":g,
        "key":jax.random.PRNGKey(0),
        "big_it":jnp.array(0,dtype=jnp.int32),
        "disorder":jnp.array(1,dtype=jnp.int32),
        "learning_curve":jnp.zeros(
            n_big_steps,
            dtype=jnp.int32,
        ),
    }

    state=jax.lax.while_loop(
        cond_fn,
        body_fn,
        init_state,
    )

    g,(lc2,last_it)=_cleanup(
        state["g"],
        points,
        end_loop,
        loop,
        loopseq,
        max_iter,
        verbose,
    )

    lc1=state["learning_curve"]

    return g,(
        jnp.concatenate([lc1,lc2]),
        last_it,
    )


# ============================================================================
# Ultimate Variant
# ============================================================================

def ultimate_carthesian_sort(gridmap, points, max_iter=100, verbose=2, loop=None, loopseq="decreasing"):
    g, dims, loop, circular_loop, end_loop = _prepare(gridmap, points, loop, loopseq)

    # Phase 1 — Robust sort
    g, (lc1, _) = robust_carthesian_sort(
        g, points,
        max_iter=max_iter, verbose=verbose,
        loop=loop, loopseq=loopseq,
    )

    # Phase 2 — HashTable refinement with big steps
    htable = HashTable(g, dims)
    circular = False
    chunk_size = 10
    
    for big_it in range((4 * max_iter) // chunk_size):
        for _ in range(chunk_size):
            for heuristic in circular_loop:
                if circular:
                    htable.gtable = heuristic[htable.gtable]
                circular = True
                htable.sort()

    g, (lc2, last_it) = _cleanup(htable.gtable, points, end_loop, loop, loopseq, max_iter, verbose)
    return g, (jnp.concatenate([lc1, lc2]), last_it)