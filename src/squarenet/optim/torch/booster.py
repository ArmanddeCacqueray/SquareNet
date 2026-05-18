import torch


def _scatter_perm(h, hp, N, device):
    out = torch.zeros(N, dtype=torch.int32, device=device)
    out[h] = hp
    return out

def integer_boost(points):
    points = torch.as_tensor(points)

    N, D = points.shape
    device = points.device

    ranks = []

    for d in range(D):
        order = torch.argsort(points[:, d])

        rank = torch.empty(N, dtype=torch.int32, device=device)
        rank[order] = torch.arange(N, dtype=torch.int32, device=device)

        ranks.append(rank)

    return tuple(ranks)


def loop_boost(points):
    int_boost = integer_boost(points)

    N = int_boost[0].shape[0]
    device = points.device

    identity = torch.arange(N, dtype=torch.int32, device=device)

    h_sources = (identity,) + int_boost
    h_targets = int_boost + (identity,)

    loop = tuple(
        _scatter_perm(h, hp, N, device)
        for h, hp in zip(h_sources, h_targets)
    )

    init_loop = loop[0]
    end_loop = loop[-1]

    circular_loop = list(loop[:-1])
    circular_loop[0] = init_loop[end_loop]

    return init_loop, tuple(circular_loop), end_loop