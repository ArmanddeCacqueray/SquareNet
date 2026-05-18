import itertools

import jax
import jax.numpy as jnp


def random_subgrid_split(key,gshape,dims):

    D=len(dims)

    keys=jax.random.split(key,D+1)

    perms=[
        jax.random.permutation(keys[i],gshape[d])
        for i,d in enumerate(dims)
    ]

    x0=[
        jnp.sort(p[:p.shape[0]//2])
        for p in perms
    ]

    x1=[
        jnp.sort(p[p.shape[0]//2:])
        for p in perms
    ]

    patterns=jnp.array(
        list(itertools.product([0,1],repeat=D)),
        dtype=jnp.int32,
    )

    perm=jax.random.permutation(
        keys[-1],
        patterns.shape[0],
    )

    patterns=patterns[perm]

    n_per_dim=(2**D)//D

    patterns=patterns[:D*n_per_dim]
    patterns=patterns.reshape(D,n_per_dim,D)

    full_axes=tuple(
        jnp.arange(s)
        for s in gshape
    )

    subgrids=[]

    for di in range(D):

        dim_subs=[]

        for dj in range(n_per_dim):

            bits=patterns[di,dj]

            idx=list(full_axes)

            for i in range(D):

                axis=dims[i]

                idx0=x0[i]
                idx1=x1[i]

                idx[axis]=jax.lax.select(
                    bits[i].astype(bool),
                    idx1,
                    idx0,
                )

            dim_subs.append(
                jnp.ix_(*idx)
            )

        subgrids.append(tuple(dim_subs))

    return tuple(subgrids)