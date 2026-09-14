"""Small renderer-order and screen-space clipping helpers.

These helpers are deliberately independent of Qt and of the hatching-depth
logic.  They only describe the order in which already prepared primitives are
painted and the visible part of a projected bond.
"""

from __future__ import annotations

import numpy as np


def screen_distance(camera_point: np.ndarray) -> float:
    """Return a signed distance proxy from the virtual screen.

    The viewer uses camera-space +Z as *nearer*.  Therefore ``-Z`` has the
    useful distance convention for painter ordering: a larger value is farther
    from the screen/camera and should be painted earlier.

    Only relative values are used, so the arbitrary additive position of the
    virtual screen is irrelevant.
    """

    point = np.asarray(camera_point, dtype=float).reshape(3)
    return -float(point[2])


def screen_distances(camera_points: np.ndarray) -> tuple[float, ...]:
    """Return screen-distance values for every supplied camera-space point."""

    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    return tuple(screen_distance(point) for point in points)




def atom_screen_distances(camera_center: np.ndarray) -> tuple[float]:
    """Distance tuple for one displayed atomic position."""

    return (screen_distance(camera_center),)


def bond_screen_distances(
    camera_first: np.ndarray,
    camera_second: np.ndarray,
) -> tuple[float, float]:
    """Distances of the two ends of the actually displayed bond segment."""

    return screen_distance(camera_first), screen_distance(camera_second)


def face_screen_distances(camera_vertices: np.ndarray) -> tuple[float, ...]:
    """Distances of every corner of one displayed polyhedron face."""

    return screen_distances(camera_vertices)


def draw_order_key(distances: tuple[float, ...]) -> tuple[float, float, float]:
    """Collapse characteristic-point distances into one stable painter key.

    Primitives are sorted with ``reverse=True``.  Therefore we compare, in
    order, the farthest characteristic point, their mean distance, and finally
    the nearest characteristic point.  Faces use all corners, bonds use their
    two *visible* ends, and atoms use their displayed position.
    """

    if not distances:
        raise ValueError("draw_order_key requires at least one distance")
    values = np.asarray(distances, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("draw-order distances must be finite")
    return float(values.max()), float(values.mean()), float(values.min())


def clip_projected_bond(
    first: np.ndarray,
    second: np.ndarray,
    first_radius: float,
    second_radius: float,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Clip a projected centre-to-centre bond to the two atom circles.

    Returns ``(visible_first, visible_second, t_first, t_second)`` where the
    ``t`` values are fractions along the original projected segment.  Because
    the viewer projection is orthographic, those same fractions can be used to
    interpolate the camera-space Z values of the actually drawn bond ends.

    If the projected atom circles overlap enough to cover the entire segment,
    there is no visible bond and ``None`` is returned.
    """

    first_xy = np.asarray(first, dtype=float).reshape(2)
    second_xy = np.asarray(second, dtype=float).reshape(2)
    radius_a = max(0.0, float(first_radius))
    radius_b = max(0.0, float(second_radius))
    direction = second_xy - first_xy
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return None
    if length <= radius_a + radius_b + 1e-9:
        return None

    unit = direction / length
    t_first = radius_a / length
    t_second = 1.0 - radius_b / length
    visible_first = first_xy + unit * radius_a
    visible_second = second_xy - unit * radius_b
    return visible_first, visible_second, t_first, t_second


def inset_polygon(
    points: np.ndarray,
    inset: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Move every polygon corner towards the polygon centre.

    ``inset`` may be either a single scalar value used for every corner or one
    value per polygon vertex.  Each shift is limited so that the point never
    crosses the centre.
    """

    polygon = np.asarray(points, dtype=float).reshape(-1, 2)
    if polygon.shape[0] == 0:
        return polygon.copy()
    inset_values = np.asarray(inset, dtype=float)
    if inset_values.ndim == 0:
        inset_values = np.full((polygon.shape[0],), float(inset_values), dtype=float)
    else:
        inset_values = inset_values.reshape(-1)
        if inset_values.shape[0] != polygon.shape[0]:
            raise ValueError("inset polygon expects one inset value per vertex")
    centre = polygon.mean(axis=0)
    shifted = polygon.copy()
    for index, point in enumerate(polygon):
        direction = centre - point
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            continue
        step = min(max(0.0, float(inset_values[index])), 0.49 * length)
        shifted[index] = point + direction * (step / length)
    return shifted

def clip_polygon_vertices_3d(
    vertices: np.ndarray,
    radii: float | np.ndarray,
) -> np.ndarray:
    """Clip polygon corners in 3D by real per-vertex radii.

    Each original vertex is replaced by two points lying on the adjacent edges,
    at the requested distance from the vertex.  This creates a chamfer-like
    corner cut in the polygon plane before any projection to screen space.
    """

    polygon = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if polygon.shape[0] < 3:
        return polygon.copy()
    radius_values = np.asarray(radii, dtype=float)
    if radius_values.ndim == 0:
        radius_values = np.full((polygon.shape[0],), float(radius_values), dtype=float)
    else:
        radius_values = radius_values.reshape(-1)
        if radius_values.shape[0] != polygon.shape[0]:
            raise ValueError("clip polygon expects one radius value per vertex")

    cut_prev = np.empty_like(polygon)
    cut_next = np.empty_like(polygon)
    for index, point in enumerate(polygon):
        previous = polygon[(index - 1) % polygon.shape[0]]
        following = polygon[(index + 1) % polygon.shape[0]]
        to_previous = previous - point
        to_following = following - point
        length_previous = float(np.linalg.norm(to_previous))
        length_following = float(np.linalg.norm(to_following))
        if length_previous <= 1e-12 or length_following <= 1e-12:
            cut_prev[index] = point
            cut_next[index] = point
            continue
        radius = max(0.0, float(radius_values[index]))
        step_previous = min(radius, 0.49 * length_previous)
        step_following = min(radius, 0.49 * length_following)
        cut_prev[index] = point + to_previous * (step_previous / length_previous)
        cut_next[index] = point + to_following * (step_following / length_following)

    result: list[np.ndarray] = []
    tolerance = 1e-12
    for index in range(polygon.shape[0]):
        first = cut_next[index]
        second = cut_prev[(index + 1) % polygon.shape[0]]
        if not result or float(np.linalg.norm(first - result[-1])) > tolerance:
            result.append(first)
        if float(np.linalg.norm(second - result[-1])) > tolerance:
            result.append(second)
    if len(result) > 1 and float(np.linalg.norm(result[0] - result[-1])) <= tolerance:
        result.pop()
    return np.asarray(result, dtype=float)

__all__ = [
    "atom_screen_distances",
    "bond_screen_distances",
    "clip_polygon_vertices_3d",
    "clip_projected_bond",
    "draw_order_key",
    "face_screen_distances",
    "inset_polygon",
    "screen_distance",
    "screen_distances",
]
