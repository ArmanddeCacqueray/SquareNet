import jax.numpy as jnp

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