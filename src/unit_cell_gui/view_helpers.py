"""Shared view algorithms used by the OpenGL unit-cell viewer.

These functions are the rendering-independent parts of the original Qt
viewer: screen-space hatch planning and camera-drag orientation math.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import QPointF

from .hatching import (
    face_hatch_segments,
    propagated_hatch_directions,
    quadrilateral_hatch_line_count,
)
from .models import Polyhedron

def _qpoint_array(point: QPointF) -> np.ndarray:
    return np.array([point.x(), point.y()], dtype=float)


def _line_direction(first: QPointF, second: QPointF) -> np.ndarray:
    direction = _qpoint_array(second) - _qpoint_array(first)
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    direction /= length
    if direction[0] < 0.0 or (
        abs(float(direction[0])) < 1e-12 and direction[1] < 0.0
    ):
        direction = -direction
    return direction


def _clip_line_to_polygon(
    polygon: list[QPointF],
    direction: np.ndarray,
    line_constant: float,
    eps: float = 1e-8,
) -> tuple[QPointF, QPointF] | None:
    """Clip an infinite parallel line against one convex screen polygon."""

    normal = np.array([-direction[1], direction[0]], dtype=float)
    points: list[np.ndarray] = []
    vertices = [_qpoint_array(point) for point in polygon]
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        first_side = float(np.dot(first, normal) - line_constant)
        second_side = float(np.dot(second, normal) - line_constant)
        if abs(first_side) < eps:
            points.append(first.copy())
        if first_side * second_side < -(eps * eps):
            fraction = first_side / (first_side - second_side)
            points.append(first + fraction * (second - first))

    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) < 1e-6 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    if len(unique) > 2:
        first, second = max(
            (
                (a, b)
                for position, a in enumerate(unique)
                for b in unique[position + 1 :]
            ),
            key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])),
        )
    else:
        first, second = unique
    return QPointF(float(first[0]), float(first[1])), QPointF(
        float(second[0]), float(second[1])
    )


def _face_adjacency(
    faces: tuple[tuple[int, ...], ...],
) -> tuple[dict[int, set[int]], dict[tuple[int, int], list[int]]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for position, first in enumerate(face):
            second = face[(position + 1) % len(face)]
            edge = tuple(sorted((int(first), int(second))))
            edge_faces.setdefault(edge, []).append(face_index)
    adjacency = {index: set() for index in range(len(faces))}
    for owners in edge_faces.values():
        for first in owners:
            adjacency[first].update(second for second in owners if second != first)
    return adjacency, edge_faces


def _longest_edge(face: tuple[int, ...], projected: list[QPointF]) -> tuple[int, int]:
    edges = [
        (int(face[index]), int(face[(index + 1) % len(face)]))
        for index in range(len(face))
    ]
    return max(
        edges,
        key=lambda edge: float(
            np.linalg.norm(_qpoint_array(projected[edge[0]]) - _qpoint_array(projected[edge[1]]))
        ),
    )


def _opposite_edge(
    face: tuple[int, ...],
    apex: int,
    projected: list[QPointF],
) -> tuple[int, int]:
    candidates = []
    for position, first in enumerate(face):
        second = face[(position + 1) % len(face)]
        if apex not in {first, second}:
            candidates.append((int(first), int(second)))
    if not candidates:
        return _longest_edge(face, projected)
    apex_point = _qpoint_array(projected[apex])
    return max(
        candidates,
        key=lambda edge: float(
            np.linalg.norm(
                0.5
                * (_qpoint_array(projected[edge[0]]) + _qpoint_array(projected[edge[1]]))
                - apex_point
            )
        ),
    )


def _is_octahedron(polyhedron: Polyhedron) -> bool:
    return (
        polyhedron.coordination_number == 6
        and len(polyhedron.faces) == 8
        and all(len(face) == 3 for face in polyhedron.faces)
    )


def _octahedron_opposite_vertex(
    polyhedron: Polyhedron,
    apex: int,
) -> int | None:
    neighbours: set[int] = set()
    for face in polyhedron.faces:
        if apex in face:
            neighbours.update(int(vertex) for vertex in face if vertex != apex)
    candidates = set(range(polyhedron.coordination_number)) - neighbours - {apex}
    return next(iter(candidates)) if len(candidates) == 1 else None


def _octahedron_face_families(
    polyhedron: Polyhedron,
    visible_faces: list[int],
    anchor: int,
) -> tuple[set[int], dict[int, int], dict[int, tuple[int, int]]]:
    """Reproduce the original three continuous families around a blank face."""

    visible = set(visible_faces)
    adjacency, edge_faces = _face_adjacency(polyhedron.faces)
    blank = {anchor}
    families: dict[int, int] = {}
    references: dict[int, tuple[int, int]] = {}
    queue: deque[int] = deque()

    anchor_face = polyhedron.faces[anchor]
    for position, first in enumerate(anchor_face):
        second = anchor_face[(position + 1) % len(anchor_face)]
        edge = tuple(sorted((int(first), int(second))))
        for neighbour in edge_faces.get(edge, []):
            if neighbour == anchor or neighbour not in visible or neighbour in families:
                continue
            family = len(references)
            references[family] = (int(first), int(second))
            families[neighbour] = family
            queue.append(neighbour)

    while queue:
        face_index = queue.popleft()
        family = families[face_index]
        for neighbour in adjacency[face_index]:
            if neighbour not in visible or neighbour in blank or neighbour in families:
                continue
            families[neighbour] = family
            queue.append(neighbour)
    return blank, families, references


def _single_face_hatches(
    face: tuple[int, ...],
    projected: list[QPointF],
    reference_edge: tuple[int, int],
    divisions: int,
) -> list[tuple[QPointF, QPointF]]:
    """Fill one face with equally spaced lines parallel to a real face edge."""

    if len(face) == 3:
        edge_vertices = {int(reference_edge[0]), int(reference_edge[1])}
        apex_candidates = [
            int(vertex) for vertex in face if int(vertex) not in edge_vertices
        ]
        if len(apex_candidates) == 1:
            apex = projected[apex_candidates[0]]
            first = projected[int(reference_edge[0])]
            second = projected[int(reference_edge[1])]
            return [
                (
                    QPointF(
                        apex.x() + fraction * (first.x() - apex.x()),
                        apex.y() + fraction * (first.y() - apex.y()),
                    ),
                    QPointF(
                        apex.x() + fraction * (second.x() - apex.x()),
                        apex.y() + fraction * (second.y() - apex.y()),
                    ),
                )
                for step in range(1, divisions + 1)
                for fraction in (step / (divisions + 1.0),)
            ]

    first, second = (projected[index] for index in reference_edge)
    direction = _line_direction(first, second)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    polygon = [projected[index] for index in face]
    values = np.asarray([float(np.dot(_qpoint_array(point), normal)) for point in polygon])
    minimum, maximum = float(values.min()), float(values.max())
    span = maximum - minimum
    if span <= 1e-9:
        return []
    constants = [
        minimum + step * span / (divisions + 1.0)
        for step in range(1, divisions + 1)
    ]
    return [
        segment
        for constant in constants
        if (segment := _clip_line_to_polygon(polygon, direction, constant)) is not None
    ]



def _polyhedron_hatches(
    polyhedron: Polyhedron,
    vertices_camera: np.ndarray,
    projected: list[QPointF],
    visible_faces: list[int],
    normals: dict[int, np.ndarray],
    divisions: int,
    project_point,
) -> dict[int, list[tuple[QPointF, QPointF]]]:
    result = {face_index: [] for face_index in visible_faces}
    if not visible_faces:
        return result

    anchor = max(visible_faces, key=lambda index: float(normals[index][2]))
    face_mode = float(normals[anchor][2]) >= 0.999
    octahedron = _is_octahedron(polyhedron)

    if not octahedron:
        blank = {anchor} if face_mode else set()
        directions = propagated_hatch_directions(
            polyhedron.faces, vertices_camera, visible_faces, blank
        )
        for face_index, direction in directions.items():
            face = polyhedron.faces[face_index]
            count = (
                quadrilateral_hatch_line_count(divisions)
                if len(face) == 4 else divisions
            )
            result[face_index] = [
                (project_point(first), project_point(second))
                for first, second in face_hatch_segments(
                    vertices_camera[list(face)], direction, count
                )
            ]
        return result

    if octahedron and face_mode:
        blank, families, references = _octahedron_face_families(
            polyhedron,
            visible_faces,
            anchor,
        )
        by_family: dict[int, list[int]] = {}
        for face_index, family in families.items():
            by_family.setdefault(family, []).append(face_index)
        for family, face_indices in by_family.items():
            edge = references[family]
            first, second = projected[edge[0]], projected[edge[1]]
            direction = _line_direction(first, second)
            normal = np.array([-direction[1], direction[0]], dtype=float)
            points = np.asarray(
                [
                    _qpoint_array(projected[index])
                    for face_index in face_indices
                    for index in polyhedron.faces[face_index]
                ],
                dtype=float,
            )
            values = points @ normal
            minimum, maximum = float(values.min()), float(values.max())
            span = maximum - minimum
            if span <= 1e-9:
                continue
            reference = 0.5 * (
                float(np.dot(_qpoint_array(first), normal))
                + float(np.dot(_qpoint_array(second), normal))
            )
            step = span / (divisions + 1.0)
            constants = [
                reference + offset * step
                for offset in range(-divisions - 2, divisions + 3)
            ]
            for face_index in face_indices:
                polygon = [projected[index] for index in polyhedron.faces[face_index]]
                for constant in constants:
                    segment = _clip_line_to_polygon(polygon, direction, constant)
                    if segment is not None:
                        result[face_index].append(segment)
        for face_index in blank:
            result[face_index] = []
        return result

    blank = {anchor} if face_mode else set()
    nearest = int(np.argmax(vertices_camera[:, 2])) if octahedron else None
    opposite = (
        _octahedron_opposite_vertex(polyhedron, nearest)
        if nearest is not None
        else None
    )
    anchor_face = polyhedron.faces[anchor]
    anchor_edges = {
        tuple(
            sorted(
                (
                    int(first),
                    int(anchor_face[(position + 1) % len(anchor_face)]),
                )
            )
        )
        for position, first in enumerate(anchor_face)
    }
    for face_index in visible_faces:
        if face_index in blank:
            continue
        face = polyhedron.faces[face_index]
        reference_edge = None
        if face_mode and not octahedron:
            for position, first in enumerate(face):
                second = face[(position + 1) % len(face)]
                if tuple(sorted((int(first), int(second)))) in anchor_edges:
                    reference_edge = (int(first), int(second))
                    break
        if reference_edge is None:
            if nearest is not None and nearest in face:
                apex = nearest
            elif opposite is not None and opposite in face:
                apex = opposite
            else:
                apex = max(face, key=lambda vertex: float(vertices_camera[vertex, 2]))
            reference_edge = _opposite_edge(face, int(apex), projected)
        result[face_index] = _single_face_hatches(
            face,
            projected,
            reference_edge,
            divisions,
        )
    return result



def _axis_rotation(axis: int, angle_degrees: float) -> np.ndarray:
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    result = np.eye(3)
    first, second = ((1, 2), (0, 2), (0, 1))[axis]
    result[first, first] = result[second, second] = cosine
    result[first, second] = -sine
    result[second, first] = sine
    return result


def screen_drag_orientation(orientation: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Rotate the displayed scene around its current screen X and Y axes."""
    return _axis_rotation(1, 0.45 * dx) @ _axis_rotation(0, -0.45 * dy) @ orientation



__all__ = ["_polyhedron_hatches", "screen_drag_orientation"]
