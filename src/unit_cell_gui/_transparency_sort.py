"""Back-to-front ordering of planar convex faces and physical line segments.

Geometry stays resident. This module only returns indices into a draw list.
Depth constraints are evaluated on projected overlap, not at polygon centres.
Intersecting faces and cyclic visibility are outside the input contract; their
count is returned for diagnostics, with a deterministic order as a fallback.
"""
from dataclasses import dataclass, replace
import heapq
import sys

import numpy as np


@dataclass
class _Projection:
    points: list
    polygon: list
    planes: list
    depth: tuple | None
    bounds: np.ndarray
    edge: bool
    key: tuple


@dataclass
class PreparedPrimitives:
    points: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    world_normals: np.ndarray
    stable_keys: tuple
    groups: tuple
    extent: float


def prepare_primitives(primitives):
    """Own camera-independent packed geometry and stable tie keys."""
    arrays = [np.asarray(points, dtype=float).reshape(-1, 3) for _, points, _ in primitives]
    counts = np.asarray([len(points) for points in arrays], dtype=int)
    starts = np.r_[0, np.cumsum(counts)[:-1]] if len(counts) else np.zeros(0, int)
    points = np.concatenate(arrays) if arrays else np.empty((0, 3))
    edges = np.asarray([edge for edge, _, _ in primitives], dtype=bool)
    normals = np.zeros((len(primitives), 3))
    faces = np.flatnonzero(~edges)
    if len(faces):
        first = starts[faces]
        normals[faces] = np.cross(points[first+1]-points[first], points[first+2]-points[first])
    keys = tuple((int(edge), material, tuple(sorted(map(tuple, array))))
                 for (edge, _, material), array in zip(primitives, arrays))
    for array in (points, starts, counts, edges, normals):
        array.setflags(write=False)
    groups = tuple((ids, starts[ids, None]+np.arange(count)[None, :])
                   for count in np.unique(counts)
                   for ids in (np.flatnonzero(counts == count),))
    for ids, point_ids in groups:
        ids.setflags(write=False)
        point_ids.setflags(write=False)
    return PreparedPrimitives(points, starts, counts, edges, normals, keys, groups,
                              max(1., float(np.max(np.abs(points)))) if len(points) else 1.)


def material_keys(prepared, materials):
    """Change stable material ties without repacking unchanged world geometry."""
    if len(materials) != len(prepared.stable_keys):
        raise ValueError('material count differs from prepared geometry')
    keys = tuple((old[0], material, old[2]) for old,material in zip(prepared.stable_keys,materials))
    return replace(prepared,stable_keys=keys)


def _project(prepared, rotation, margin, bias):
    points = prepared.points @ rotation.T
    if bias:
        points[np.repeat(prepared.edges, prepared.counts), 2] += bias
    low = np.minimum.reduceat(points[:, :2], prepared.starts)
    high = np.maximum.reduceat(points[:, :2], prepared.starts)
    padding = prepared.edges.astype(float)*margin
    bounds = np.column_stack((low-padding[:, None], high+padding[:, None]))
    # Keep the previous per-polygon NumPy mean arithmetic for stable ties;
    # reduceat uses a different summation order and can swap coplanar faces.
    means = np.empty(len(prepared.counts))
    for ids, point_ids in prepared.groups:
        means[ids] = points[point_ids, 2].mean(axis=1)
    # Cross products transform with the cofactor matrix, including reflection
    # and non-orthogonal invertible orientations. No rotation-only assumption.
    cofactor = np.array([np.cross(rotation[1], rotation[2]),
                         np.cross(rotation[2], rotation[0]),
                         np.cross(rotation[0], rotation[1])])
    normals = prepared.world_normals @ cofactor.T
    scalar = points.tolist()
    epsilon = prepared.extent*1e-10
    projected = []
    for i, (start, count) in enumerate(zip(prepared.starts, prepared.counts)):
        row = scalar[start:start+count]
        edge = bool(prepared.edges[i])
        polygon = [tuple(point[:2]) for point in row] if not edge else []
        depth = None
        if not edge and abs(normals[i, 2]) > epsilon:
            a, b = (float(value) for value in -normals[i, :2]/normals[i, 2])
            depth = (a, b, row[0][2]-a*row[0][0]-b*row[0][1])
        key = (float(means[i]),)+prepared.stable_keys[i]
        projected.append(_Projection(row, polygon, _halfplanes(polygon) if not edge else [],
                                      depth, bounds[i], edge, key))
    return projected, bounds


def _halfplanes(points):
    area = sum(p[0]*q[1]-p[1]*q[0] for p,q in zip(points,points[1:]+points[:1]))
    direction = 1 if area >= 0 else -1
    return [(direction*(p[1]-q[1]),direction*(q[0]-p[0]),
             direction*(p[0]*q[1]-p[1]*q[0])) for p,q in zip(points,points[1:]+points[:1])]


def _clip_polygon(subject, planes, epsilon):
    for a,b,c in planes:
        if not subject:
            break
        result = []
        previous = subject[-1]
        d0 = a*previous[0]+b*previous[1]+c
        for current in subject:
            d1 = a*current[0]+b*current[1]+c
            if (d0 >= -epsilon) != (d1 >= -epsilon):
                fraction = d0/(d0-d1)
                result.append((previous[0]+fraction*(current[0]-previous[0]),
                               previous[1]+fraction*(current[1]-previous[1])))
            if d1 >= -epsilon:
                result.append(current)
            previous,d0 = current,d1
        subject = result
    return subject


def _depth_at(item, x, y):
    a,b,c = item.depth
    return a*x+b*y+c


@dataclass
class _BoundsNode:
    bounds: np.ndarray
    ids: np.ndarray | None = None
    left: object = None
    right: object = None


def _overlapping_pair_batches(bounds, epsilon):
    """Yield each overlapping pair once, using a balanced 2D bounds hierarchy.

    Leaves contain at most 32 primitives. Traversal rejects disjoint node
    pairs before any primitive tests; it never constructs an N by N array.
    Dense projected overlap can still produce quadratically many candidates.
    """
    if len(bounds) < 2:
        return

    def build(ids):
        boxes = bounds[ids]
        envelope = np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]
        if len(ids) <= 32:
            return _BoundsNode(envelope, ids=ids)
        centres = boxes[:, :2] * .5 + boxes[:, 2:] * .5
        axis = int(np.argmax(np.ptp(centres, axis=0)))
        middle = len(ids) // 2
        order = np.argpartition(centres[:, axis], middle)
        return _BoundsNode(envelope, left=build(ids[order[:middle]]),
                           right=build(ids[order[middle:]]))

    root = build(np.arange(len(bounds)))
    stack = [(root, root)]
    while stack:
        first, second = stack.pop()
        a, b = first.bounds, second.bounds
        if (a[0] > b[2] + epsilon or b[0] > a[2] + epsilon
                or a[1] > b[3] + epsilon or b[1] > a[3] + epsilon):
            continue
        if first is second and first.ids is None:
            stack.extend(((first.left, first.left), (first.left, first.right),
                          (first.right, first.right)))
        elif first.ids is not None and second.ids is not None:
            a, b = bounds[first.ids], bounds[second.ids]
            overlap = ((a[:, None, 0] <= b[None, :, 2] + epsilon)
                       & (b[None, :, 0] <= a[:, None, 2] + epsilon)
                       & (a[:, None, 1] <= b[None, :, 3] + epsilon)
                       & (b[None, :, 1] <= a[:, None, 3] + epsilon))
            if first is second:
                overlap &= np.triu(np.ones(overlap.shape, dtype=bool), 1)
            rows, columns = np.nonzero(overlap)
            if len(rows):
                pairs = np.column_stack((first.ids[rows], second.ids[columns]))
                pairs.sort(axis=1)
                yield pairs
        elif second.ids is not None or (first.ids is None
                and np.max(first.bounds[2:] - first.bounds[:2])
                >= np.max(second.bounds[2:] - second.bounds[:2])):
            stack.extend(((first.left, second), (first.right, second)))
        else:
            stack.extend(((first, second.left), (first, second.right)))


def _line_face_constraints(projected, edge_ids, face_ids, epsilon,
                           depth_epsilon, following, incoming):
    """Clip only candidate edge/face pairs, in bounded NumPy batches.

    Each row is one overlapping pair, rather than an edge crossed with every
    face. Padded halfplanes are zero and impose no additional constraint.
    """
    count = max(len(projected[i].planes) for i in face_ids)
    batch_size = max(1, 65536 // count)
    conflicts = 0
    for start in range(0, len(edge_ids), batch_size):
        edges, faces = edge_ids[start:start+batch_size], face_ids[start:start+batch_size]
        # A face/edge commonly participates in many candidate pairs. Convert
        # each distinct primitive once per batch, then gather paired rows.
        unique_faces, face_rows = np.unique(faces, return_inverse=True)
        planes = np.zeros((len(unique_faces), count, 3))
        for row, i in enumerate(unique_faces):
            planes[row, :len(projected[i].planes)] = projected[i].planes
        planes = planes[face_rows]
        depths = np.array([projected[i].depth for i in unique_faces])[face_rows]
        unique_edges, edge_rows = np.unique(edges, return_inverse=True)
        points = np.array([projected[i].points for i in unique_edges])[edge_rows]
        p, direction = points[:, 0], points[:, 1] - points[:, 0]
        distance = (p[:, None, 0]*planes[:, :, 0]
                    + p[:, None, 1]*planes[:, :, 1] + planes[:, :, 2])
        delta = (direction[:, None, 0]*planes[:, :, 0]
                 + direction[:, None, 1]*planes[:, :, 1])
        parallel = np.abs(delta) <= epsilon
        crossing = np.divide(-distance, delta, out=np.zeros_like(distance), where=~parallel)
        low = np.maximum(0., np.max(np.where(delta > epsilon, crossing, 0.), axis=1))
        high = np.minimum(1., np.min(np.where(delta < -epsilon, crossing, 1.), axis=1))
        valid = ~np.any(parallel & (distance < -epsilon), axis=1) & (low <= high+epsilon)
        first = p[:, 2] - (p[:, 0]*depths[:, 0] + p[:, 1]*depths[:, 1] + depths[:, 2])
        step = direction[:, 2] - (direction[:, 0]*depths[:, 0] + direction[:, 1]*depths[:, 1])
        d0, d1 = first + low*step, first + high*step
        lower, upper = np.minimum(d0, d1), np.maximum(d0, d1)
        for mask, edge_behind in ((valid & (upper <= depth_epsilon) & (lower < -depth_epsilon), True),
                                 (valid & (lower >= -depth_epsilon) & (upper > depth_epsilon), False)):
            for row in np.flatnonzero(mask):
                edge_id, face_id = int(edges[row]), int(faces[row])
                behind, front = (edge_id, face_id) if edge_behind else (face_id, edge_id)
                following[behind].add(front)
                incoming[front] += 1
        conflicts += int(np.count_nonzero(valid & (lower < -depth_epsilon) & (upper > depth_epsilon)))
    return conflicts


def _differences(first, second, epsilon, margin):
    """Depth(first)-depth(second) at the extremities of projected overlap."""
    if not first.edge and not second.edge:
        if first.depth is None or second.depth is None:
            return ()
        overlap = _clip_polygon(first.polygon,second.planes,epsilon)
        return tuple(_depth_at(first,x,y)-_depth_at(second,x,y) for x,y in overlap)
    p,q = first.points
    r,s = second.points
    dx,dy = q[0]-p[0],q[1]-p[1]
    ex,ey = s[0]-r[0],s[1]-r[1]
    cross = dx*ey-dy*ex
    if abs(cross) > epsilon:
        t = ((r[0]-p[0])*ey-(r[1]-p[1])*ex)/cross
        u = ((r[0]-p[0])*dy-(r[1]-p[1])*dx)/cross
        if -epsilon <= t <= 1+epsilon and -epsilon <= u <= 1+epsilon:
            return (p[2]+t*(q[2]-p[2])-r[2]-u*(s[2]-r[2]),)
        return ()
    # Collinear projected edges, including coincident shared edges.
    length = (dx*dx+dy*dy)**.5
    if length <= epsilon or abs((r[0]-p[0])*dy-(r[1]-p[1])*dx) > margin*length+epsilon:
        return ()
    axis = 0 if abs(dx) >= abs(dy) else 1
    low = max(min(p[axis],q[axis]),min(r[axis],s[axis]))
    high = min(max(p[axis],q[axis]),max(r[axis],s[axis]))
    if low > high+epsilon or abs(s[axis]-r[axis]) <= epsilon:
        return ()
    return tuple(p[2]+(x-p[axis])/(q[axis]-p[axis])*(q[2]-p[2])
                 -r[2]-(x-r[axis])/(s[axis]-r[axis])*(s[2]-r[2]) for x in (low,high))


def back_to_front(primitives, orientation, *, line_margin=0., line_depth_bias=0., stats=None):
    """Return (ordered indices, unresolved count); geometry may be prepared once.

    Larger camera Z is nearer, matching the viewer's reversed depth convention.
    Optional stats describe candidates and retained graph storage, not a cap.
    """
    prepared = primitives if isinstance(primitives, PreparedPrimitives) else prepare_primitives(primitives)
    if stats is not None:
        stats.clear()
        stats.update(primitives=len(prepared.counts), candidates=0, constraints=0, graph_bytes_estimate=0)
    if not len(prepared.counts):
        return [], 0
    rotation = np.asarray(orientation, dtype=float)
    epsilon = prepared.extent*1e-10
    depth_epsilon = prepared.extent*1e-7
    projected, bounds = _project(prepared, rotation, line_margin, line_depth_bias)
    following = [set() for _ in projected]
    incoming = np.zeros(len(projected),dtype=int)
    unresolved = 0
    is_edge = np.array([item.edge for item in projected])
    has_depth = np.array([item.depth is not None for item in projected])
    pending_pairs, pending_count = [], 0

    def process_pairs(pairs):
        conflicts = 0
        first_ids, second_ids = pairs.T
        mixed = is_edge[first_ids] != is_edge[second_ids]
        if np.any(mixed):
            mixed_first, mixed_second = first_ids[mixed], second_ids[mixed]
            edges = np.where(is_edge[mixed_first], mixed_first, mixed_second)
            faces = np.where(is_edge[mixed_first], mixed_second, mixed_first)
            valid = has_depth[faces]
            if np.any(valid):
                conflicts += _line_face_constraints(projected, edges[valid], faces[valid],
                    epsilon, depth_epsilon, following, incoming)
        for i, j in pairs[~mixed].tolist():
            difference = _differences(projected[i], projected[j], epsilon, line_margin)
            if not difference:
                continue
            low, high = min(difference), max(difference)
            if high <= depth_epsilon and low < -depth_epsilon:
                behind, front = i, j
            elif low >= -depth_epsilon and high > depth_epsilon:
                behind, front = j, i
            elif low < -depth_epsilon and high > depth_epsilon:
                conflicts += 1
                continue
            else:
                continue
            following[behind].add(front)
            incoming[front] += 1
        return conflicts

    for pairs in _overlapping_pair_batches(bounds, epsilon):
        if stats is not None:
            stats['candidates'] += len(pairs)
        pending_pairs.append(pairs)
        pending_count += len(pairs)
        if pending_count >= 4096:
            unresolved += process_pairs(np.concatenate(pending_pairs))
            pending_pairs.clear()
            pending_count = 0
    if pending_pairs:
        unresolved += process_pairs(np.concatenate(pending_pairs))
    if stats is not None:
        constraints = sum(len(row) for row in following)
        stats['constraints'] = constraints
        stats['graph_bytes_estimate'] = (sys.getsizeof(following)
            + sum(sys.getsizeof(row) for row in following) + constraints*sys.getsizeof(0)
            + incoming.nbytes)
    ready = [(item.key,i) for i,item in enumerate(projected) if incoming[i]==0]
    heapq.heapify(ready)
    result = []
    emitted = set()
    while len(result) < len(projected):
        if not ready:
            # Invalid cyclic visibility must neither hang the GUI nor trigger
            # a hidden peeling path. Keep a deterministic diagnostic fallback.
            i = min((i for i in range(len(projected)) if i not in emitted),
                    key=lambda i:projected[i].key)
            heapq.heappush(ready,(projected[i].key,i))
            unresolved += 1
        _,i = heapq.heappop(ready)
        if i in emitted:
            continue
        emitted.add(i)
        result.append(i)
        for j in following[i]:
            incoming[j] -= 1
            if incoming[j]==0:
                heapq.heappush(ready,(projected[j].key,j))
    return result,unresolved
