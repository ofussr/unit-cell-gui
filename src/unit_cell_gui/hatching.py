"""View-dependent hatching used by the unit-cell renderer."""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


def occupancy_fractions(components: Iterable[object]) -> tuple[float, ...]:
    """Return drawable site fractions while preserving partial occupancy."""
    occupancies = tuple(max(0.0, float(component.occupancy)) for component in components)
    total = sum(occupancies)
    scale = 1.0 if total <= 1.0 else 1.0 / total
    return tuple(occupancy * scale for occupancy in occupancies)


def quadrilateral_hatch_line_count(triangle_count: int) -> int:
    """Return the number of full, parallel strokes on a quadrilateral."""
    return 2 * max(0, int(triangle_count)) + 1


def _hatch_face_normal(points: np.ndarray) -> np.ndarray:
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    return normal / np.linalg.norm(normal)


def _initial_hatch_direction(face: tuple[int, ...], vertices: np.ndarray) -> np.ndarray:
    apex = max(face, key=lambda vertex: float(vertices[vertex, 2]))
    if len(face) == 4:
        # Virtual triangles choose a diagonal direction only. They never
        # become visible boundaries or separately rendered subfaces.
        position = face.index(apex)
        first, second = face[(position - 1) % 4], face[(position + 1) % 4]
    else:
        candidates = [
            (first, face[(position + 1) % len(face)])
            for position, first in enumerate(face)
            if apex not in (first, face[(position + 1) % len(face)])
        ]
        first, second = max(
            candidates,
            key=lambda edge: float(np.linalg.norm(
                0.5 * (vertices[edge[0], :2] + vertices[edge[1], :2])
                - vertices[apex, :2]
            )),
        )
    direction = vertices[second] - vertices[first]
    return direction / np.linalg.norm(direction)


def _transport_hatch_direction(
    direction: np.ndarray,
    edge: np.ndarray,
    source_normal: np.ndarray,
    target_normal: np.ndarray,
) -> np.ndarray:
    """Unfold and refold a tangent vector about the actual shared edge."""
    along = edge / np.linalg.norm(edge)
    source_across = np.cross(source_normal, along)
    target_across = np.cross(target_normal, along)
    transported = (
        float(np.dot(direction, along)) * along
        + float(np.dot(direction, source_across)) * target_across
    )
    return transported / np.linalg.norm(transported)


def propagated_hatch_directions(
    faces: Iterable[Iterable[int]],
    vertices_camera: np.ndarray,
    visible_faces: Iterable[int],
    blank_faces: Iterable[int] = (),
) -> dict[int, np.ndarray]:
    """Choose seed directions, then transport them in each receiving plane."""
    faces = tuple(tuple(int(vertex) for vertex in face) for face in faces)
    vertices = np.asarray(vertices_camera, dtype=float).reshape(-1, 3)
    visible = {int(face_index) for face_index in visible_faces}
    blank = visible.intersection(int(face_index) for face_index in blank_faces)
    if not visible:
        return {}

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        edges = tuple(
            tuple(sorted((face[position], face[(position + 1) % len(face)])))
            for position in range(len(face))
        )
        for edge in edges:
            edge_faces.setdefault(edge, []).append(face_index)

    adjacency = {face_index: set() for face_index in visible}
    shared_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for edge, owners in edge_faces.items():
        visible_owners = [face_index for face_index in owners if face_index in visible]
        for first in visible_owners:
            for second in visible_owners:
                if first == second:
                    continue
                adjacency[first].add(second)
                shared_edges[(first, second)] = edge

    depths = {
        face_index: float(vertices[list(faces[face_index]), 2].mean())
        for face_index in visible
    }
    order = sorted(visible, key=lambda face_index: (-depths[face_index], face_index))
    visible_vertices = {vertex for index in visible for vertex in faces[index]}
    nearest_depth = max(float(vertices[index, 2]) for index in visible_vertices)
    depth_tolerance = max(1e-12, float(np.ptp(vertices[:, 2])) * 1e-9)
    nearest_vertices = {
        index for index in visible_vertices
        if nearest_depth - float(vertices[index, 2]) <= depth_tolerance
    }
    normals = {
        face_index: _hatch_face_normal(vertices[list(faces[face_index])])
        for face_index in visible
    }
    directions: dict[int, np.ndarray] = {}
    for face_index in order:
        if face_index in blank:
            continue
        blank_neighbours = adjacency[face_index].intersection(blank)
        if blank_neighbours:
            parent = min(blank_neighbours)
            first, second = shared_edges[(face_index, parent)]
            direction = vertices[second] - vertices[first]
            directions[face_index] = direction / np.linalg.norm(direction)
            continue
        if nearest_vertices.intersection(faces[face_index]):
            directions[face_index] = _initial_hatch_direction(faces[face_index], vertices)
            continue
        processed_neighbours = [
            neighbour
            for neighbour in adjacency[face_index]
            if neighbour in directions
            and depths[neighbour] > depths[face_index] + depth_tolerance
        ]
        if processed_neighbours:
            parent = max(
                processed_neighbours,
                key=lambda neighbour: (depths[neighbour], -neighbour),
            )
            first, second = shared_edges[(face_index, parent)]
            directions[face_index] = _transport_hatch_direction(
                directions[parent],
                vertices[second] - vertices[first],
                normals[parent],
                normals[face_index],
            )
            continue
        directions[face_index] = _initial_hatch_direction(faces[face_index], vertices)
    return directions


def face_hatch_segments(
    face_vertices: np.ndarray,
    direction: np.ndarray,
    line_count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Clip full parallel strokes to a convex face in its own 3-D plane."""
    points = np.asarray(face_vertices, dtype=float)
    if len(points) < 3 or line_count <= 0:
        return []
    origin = points[0]
    normal = _hatch_face_normal(points)
    across = np.cross(normal, np.asarray(direction, dtype=float))
    length = float(np.linalg.norm(across))
    if length < 1e-12:
        return []
    across /= length
    values = (points - origin) @ across
    lower, upper = float(values.min()), float(values.max())
    extent = float(np.max(np.linalg.norm(points - origin, axis=1)))
    tolerance = max(1e-12, extent * 1e-10)
    if upper - lower <= tolerance:
        return []
    segments = []
    for step in range(1, int(line_count) + 1):
        constant = lower + step * (upper - lower) / (line_count + 1.0)
        crossings = []
        for index, first in enumerate(points):
            next_index = (index + 1) % len(points)
            first_side = values[index] - constant
            second_side = values[next_index] - constant
            if abs(first_side) <= tolerance:
                crossings.append(first.copy())
            if first_side * second_side < 0.0:
                fraction = first_side / (first_side - second_side)
                crossings.append(first + fraction * (points[next_index] - first))
        unique = []
        for crossing in crossings:
            if not any(np.linalg.norm(crossing - other) <= tolerance for other in unique):
                unique.append(crossing)
        if len(unique) >= 2:
            first, second = max(
                itertools.combinations(unique, 2),
                key=lambda pair: float(np.linalg.norm(pair[1] - pair[0])),
            )
            if np.linalg.norm(second - first) > tolerance:
                segments.append((first, second))
    return segments


def hatch_division_count(
    far_factor: float,
    density_setting: int,
    depth_setting: int,
) -> int:
    """Return the original density with a separate GRIP depth coefficient."""
    density_norm = (float(density_setting) - 4.0) / (42.0 - 4.0)
    density_norm = max(0.0, min(1.0, density_norm))
    near_count = 4.0 + 8.0 * density_norm
    base_difference = 4.0 + 10.0 * density_norm
    far_count = near_count + base_difference * (float(depth_setting) / 50.0)
    depth = max(0.0, min(1.0, float(far_factor)))
    smooth_depth = depth * depth * (3.0 - 2.0 * depth)
    count = near_count + (far_count - near_count) * smooth_depth
    return max(1, int(round(count)))


__all__ = [
    "face_hatch_segments",
    "hatch_division_count",
    "occupancy_fractions",
    "propagated_hatch_directions",
    "quadrilateral_hatch_line_count",
]
