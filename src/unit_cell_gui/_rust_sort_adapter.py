"""Native exact ordering, sharing the reference's float64 projection and tie keys."""
import numpy as np

try:
    from . import _ucg_native
except ImportError:
    # The standalone development wheel exposes the same extension at top level.
    import _ucg_native


def back_to_front_native(prepared, orientation, *, line_margin=0., line_depth_bias=0., stats=None):
    n = len(prepared.counts)
    if stats is not None:
        stats.clear()
        stats.update(primitives=n, candidates=0, constraints=0, graph_bytes_estimate=0)
    if not n:
        return [], 0
    rotation = np.asarray(orientation, dtype=float)
    points = prepared.points @ rotation.T
    if line_depth_bias:
        points[np.repeat(prepared.edges, prepared.counts), 2] += line_depth_bias
    means = np.empty(n)
    for ids, point_ids in prepared.groups:
        means[ids] = points[point_ids, 2].mean(axis=1)
    cofactor = np.array([np.cross(rotation[1], rotation[2]),
                         np.cross(rotation[2], rotation[0]),
                         np.cross(rotation[0], rotation[1])])
    normals = prepared.world_normals @ cofactor.T
    # Python material keys may be heterogeneous objects; preserve the exact
    # reference comparison by sending their sorted ranks, never stringifying.
    key_order = sorted(range(n), key=lambda i: (float(means[i]),) + prepared.stable_keys[i])
    ranks = np.empty(n, dtype='<u8')
    ranks[key_order] = np.arange(n, dtype='<u8')
    packed = lambda array, dtype: np.ascontiguousarray(array, dtype=dtype).tobytes()
    order, unresolved, candidates, constraints, graph_bytes = _ucg_native.sort_packed(
        packed(points, '<f8'), packed(prepared.counts, '<u8'),
        packed(prepared.edges, 'u1'), packed(normals, '<f8'),
        packed(ranks, '<u8'), prepared.extent, line_margin,
    )
    if stats is not None:
        stats.update(candidates=candidates, constraints=constraints,
                     graph_bytes_estimate=graph_bytes)
    return order, unresolved
