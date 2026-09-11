"""Fast native Qt renderer for caller-supplied unit-cell geometry."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QPainter,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .hatching import (
    hatch_division_count,
    occupancy_fractions,
    face_hatch_segments,
    propagated_hatch_directions,
    quadrilateral_hatch_line_count,
)
from .models import DisplayOptions, Polyhedron, Scene, StyleOverrides


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


def _blend_components(components) -> QColor:
    weighted = np.zeros(3, dtype=float)
    total = 0.0
    for component in components:
        colour = QColor(component.colour)
        weight = max(0.0, float(component.occupancy))
        weighted += weight * np.asarray((colour.red(), colour.green(), colour.blue()))
        total += weight
    if total <= 1e-12:
        return QColor("#b0b0b0")
    red, green, blue = np.clip(np.rint(weighted / total), 0, 255).astype(int)
    return QColor(int(red), int(green), int(blue))


def _engraving_colour(element: str, colour_value: str) -> QColor:
    if element == "O":
        return QColor(248, 248, 248)
    colour = QColor(colour_value)
    grey = int(round(0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()))
    grey = max(55, min(220, grey))
    return QColor(grey, grey, grey)


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


class UnitCellViewer(QWidget):
    """Native Qt viewport with no file loading or crystallographic calculations."""

    orientation_changed = Signal(object)
    interaction_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.scene: Scene | None = None
        self.orientation = np.eye(3)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_atoms = True
        self.show_external_atoms = True
        self.show_bonds = True
        self.show_cell = True
        self.show_basis = True
        self.show_polyhedra = False
        self.engraving = False
        self.hatching = True
        self.hatch_density = 22
        self.hatch_grip = 50
        self.atom_scale = 1.0
        self.rotation_enabled = True
        self.empty_text = "No structure is loaded"
        self.title_text = ""
        self.atom_component_visibility: dict[str, bool] = {}
        self.atom_component_colours: dict[str, str] = {}
        self.polyhedron_site_visibility: dict[str, bool] = {}
        self.polyhedron_site_colours: dict[str, str] = {}
        self.polyhedron_opaque_sites: set[str] = set()
        self._last_mouse: QPointF | None = None
        self._drag_button = None
        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        return QSize(760, 620)

    def set_scene(self, scene: Scene | None, *, reset_camera: bool = True) -> None:
        if scene is not None and not isinstance(scene, Scene):
            raise TypeError("scene must be a unit_cell_gui.Scene or None")
        self.scene = scene
        if reset_camera:
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
        self.update()

    def set_orientation(self, orientation: np.ndarray, *, emit: bool = False) -> None:
        matrix = np.asarray(orientation, dtype=float).reshape(3, 3).copy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError("orientation must contain finite values")
        self.orientation = matrix
        self.update()
        if emit:
            self.orientation_changed.emit(self.orientation.copy())

    def set_display_options(
        self,
        options: DisplayOptions | None = None,
        **changes,
    ) -> None:
        """Apply a complete option set or selected keyword changes."""
        values = {
            name: getattr(options, name) if options is not None else getattr(self, name)
            for name in DisplayOptions.__dataclass_fields__
        }
        unknown = set(changes).difference(values)
        if unknown:
            raise TypeError(f"Unknown display options: {', '.join(sorted(unknown))}")
        values.update(changes)
        normalized = DisplayOptions(**values)
        for name in DisplayOptions.__dataclass_fields__:
            setattr(self, name, getattr(normalized, name))
        self.update()

    def set_labels(self, *, empty: str | None = None, title: str | None = None) -> None:
        if empty is not None:
            self.empty_text = str(empty)
        if title is not None:
            self.title_text = str(title)
        self.update()

    def set_style_overrides(self, overrides: StyleOverrides | None = None) -> None:
        """Replace all per-component and per-polyhedron presentation state."""
        normalized = overrides or StyleOverrides()
        if not isinstance(normalized, StyleOverrides):
            raise TypeError("overrides must be unit_cell_gui.StyleOverrides or None")
        self.atom_component_visibility = dict(normalized.atom_component_visibility)
        self.atom_component_colours = dict(normalized.atom_component_colours)
        self.polyhedron_site_visibility = dict(normalized.polyhedron_site_visibility)
        self.polyhedron_site_colours = dict(normalized.polyhedron_site_colours)
        self.polyhedron_opaque_sites = set(normalized.polyhedron_opaque_sites)
        self.update()

    def zoom_by(self, steps: float) -> None:
        self.zoom = float(np.clip(self.zoom * (1.12 ** float(steps)), 0.25, 4.5))
        self.update()

    def _project(self, point: np.ndarray) -> tuple[QPointF, float]:
        width, height = max(1, self.width()), max(1, self.height())
        radius = self.scene.base_radius if self.scene is not None else 1.0
        scale = 0.88 * min(width, height) / (2.0 * radius) * self.zoom
        return (
            QPointF(
                width * 0.5 + float(point[0]) * scale + self.pan_x,
                height * 0.5 - float(point[1]) * scale + self.pan_y,
            ),
            scale,
        )

    def _component_visible(self, component) -> bool:
        return self.atom_component_visibility.get(component.key, True)

    def _component_colour(self, component) -> QColor:
        if self.engraving:
            return _engraving_colour(component.element, component.colour)
        return QColor(
            self.atom_component_colours.get(
                component.key,
                component.colour,
            )
        )

    def _atom_visible(self, atom) -> bool:
        return any(self._component_visible(component) for component in atom.components)

    def _atom_radius(self, atom) -> float:
        weighted = 0.0
        total = 0.0
        for component in atom.components:
            occupancy = max(0.0, float(component.occupancy))
            weighted += occupancy * component.radius
            total += occupancy
        radius = weighted / total if total > 1e-12 else 0.22
        return radius * self.atom_scale

    def _dominant_atom_colour(self, atom) -> QColor:
        visible = [
            component
            for component in atom.components
            if self._component_visible(component) and component.occupancy > 0.0
        ]
        if not visible:
            return QColor("#8b8b8b")
        maximum = max(float(component.occupancy) for component in visible)
        dominant = [
            component
            for component in visible
            if abs(float(component.occupancy) - maximum) < 1e-12
        ]
        if len(dominant) == 1:
            return self._component_colour(dominant[0])
        colours = [self._component_colour(component) for component in dominant]
        return QColor(
            int(round(sum(colour.red() for colour in colours) / len(colours))),
            int(round(sum(colour.green() for colour in colours) / len(colours))),
            int(round(sum(colour.blue() for colour in colours) / len(colours))),
        )

    def _mixed_atom_colour(self, atom) -> QColor:
        components = [
            component
            for component in atom.components
            if self._component_visible(component) and component.occupancy > 0.0
        ]
        total = sum(float(component.occupancy) for component in components)
        if total <= 1e-12:
            return QColor("#8b8b8b")
        channels = np.zeros(3, dtype=float)
        for component in components:
            colour = self._component_colour(component)
            channels += float(component.occupancy) * np.asarray(
                (colour.red(), colour.green(), colour.blue()),
                dtype=float,
            )
        red, green, blue = np.clip(np.rint(channels / total), 0, 255).astype(int)
        return QColor(int(red), int(green), int(blue))

    def _atom_primitive(self, atom, camera: np.ndarray) -> dict:
        point, scale = self._project(camera)
        occupancy = sum(
            max(0.0, float(component.occupancy))
            for component in atom.components
        )
        return {
            "kind": "atom",
            "depth": float(camera[2]),
            "position": point,
            "radius": max(2.5, self._atom_radius(atom) * scale),
            "components": tuple(
                {
                    "colour": self._component_colour(component),
                    "element": component.element,
                    "fraction": fraction,
                    "visible": self._component_visible(component),
                }
                for component, fraction in zip(
                    atom.components,
                    occupancy_fractions(atom.components),
                )
            ),
            "outline": self._dominant_atom_colour(atom),
            "monochrome": self._mixed_atom_colour(atom),
            "partial": occupancy < 1.0 - 1e-8,
        }

    def _polyhedron_primitives(self) -> list[dict]:
        if (
            self.scene is None
            or not self.show_polyhedra
        ):
            return []
        selected = []
        for polyhedron in self.scene.polyhedra:
            if not self.polyhedron_site_visibility.get(polyhedron.site_key, True):
                continue
            center_camera = self.orientation @ polyhedron.center
            selected.append((polyhedron, center_camera))
        if not selected:
            return []
        depths = [float(center[2]) for _polyhedron, center in selected]
        minimum, maximum = min(depths), max(depths)
        primitives = []
        for polyhedron, center_camera in selected:
            vertices_camera = np.asarray(
                [self.orientation @ vertex for vertex in polyhedron.vertices], dtype=float
            )
            projected = [self._project(vertex)[0] for vertex in vertices_camera]
            normals: dict[int, np.ndarray] = {}
            visible_faces = []
            for face_index, face in enumerate(polyhedron.faces):
                vertices = vertices_camera[list(face)]
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                length = float(np.linalg.norm(normal))
                if length < 1e-12:
                    continue
                normal /= length
                normals[face_index] = normal
                if float(normal[2]) > 1e-9:
                    visible_faces.append(face_index)
            far = (
                0.5
                if abs(maximum - minimum) < 1e-12
                else (maximum - float(center_camera[2])) / (maximum - minimum)
            )
            hatches: dict[int, list[tuple[QPointF, QPointF]]] = {}
            if self.engraving and self.hatching and visible_faces:
                hatches = _polyhedron_hatches(
                    polyhedron,
                    vertices_camera,
                    projected,
                    visible_faces,
                    normals,
                    hatch_division_count(
                        far,
                        self.hatch_density,
                        self.hatch_grip,
                    ),
                    lambda point: self._project(point)[0],
                )
            colour = QColor(
                self.polyhedron_site_colours.get(
                    polyhedron.site_key,
                    _blend_components(polyhedron.components).name(),
                )
            )
            occupancy = min(
                1.0,
                sum(max(0.0, component.occupancy) for component in polyhedron.components),
            )
            for face_index in visible_faces:
                face = polyhedron.faces[face_index]
                primitives.append(
                    {
                        "kind": "face",
                        "depth": float(vertices_camera[list(face), 2].mean()),
                        "polygon": [projected[index] for index in face],
                        "hatches": hatches.get(face_index, []),
                        "colour": colour,
                        "occupancy": occupancy,
                        "opaque": (
                            polyhedron.site_key in self.polyhedron_opaque_sites
                        ),
                    }
                )
        return primitives

    def _primitives(self) -> list[dict]:
        if self.scene is None:
            return []
        primitives = self._polyhedron_primitives()
        camera_atoms = self.scene.atom_centers @ self.orientation.T
        if self.show_bonds:
            for first, second in self.scene.bonds:
                atom_a, atom_b = self.scene.atoms[first], self.scene.atoms[second]
                if (
                    (atom_a.external or atom_b.external)
                    and not self.show_external_atoms
                ):
                    continue
                if not self._atom_visible(atom_a) or not self._atom_visible(atom_b):
                    continue
                endpoints = camera_atoms[[first, second]]
                primitives.append(
                    {
                        "kind": "bond",
                        "depth": float(endpoints[:, 2].mean()),
                        "first": self._project(endpoints[0])[0],
                        "second": self._project(endpoints[1])[0],
                        "first_colour": self._dominant_atom_colour(atom_a),
                        "second_colour": self._dominant_atom_colour(atom_b),
                    }
                )
        if self.show_atoms:
            primitives.extend(
                self._atom_primitive(atom, camera)
                for atom, camera in zip(self.scene.atoms, camera_atoms)
                if self._atom_visible(atom)
                and (self.show_external_atoms or not atom.external)
            )
        primitives.sort(key=lambda primitive: primitive["depth"])
        return primitives

    def _draw_face(self, painter: QPainter, primitive: dict) -> None:
        polygon = QPolygonF(primitive["polygon"])
        if self.engraving:
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(polygon)
            hatch_pen = QPen(QColor(30, 30, 30))
            hatch_pen.setWidthF(0.75)
            hatch_pen.setCosmetic(True)
            painter.setPen(hatch_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for first, second in primitive["hatches"]:
                painter.drawLine(first, second)
            edge = QPen(QColor(20, 20, 20))
        else:
            fill = QColor(primitive["colour"])
            if primitive["opaque"]:
                fill.setAlpha(255)
            else:
                fill.setAlpha(int(round(70 + 100 * primitive["occupancy"])))
            painter.setBrush(QBrush(fill))
            edge = QPen(fill.lighter(145))
        edge.setWidthF(1.575)
        edge.setCosmetic(True)
        painter.setPen(edge)
        painter.drawPolygon(polygon)

    def _draw_atom(self, painter: QPainter, primitive: dict) -> None:
        radius = primitive["radius"]
        point = primitive["position"]
        bounds = QRectF(
            point.x() - radius,
            point.y() - radius,
            2.0 * radius,
            2.0 * radius,
        )
        components = primitive["components"]
        if self.engraving:
            painter.setBrush(QBrush(QColor(primitive["monochrome"])))
            pen = QPen(QColor(30, 30, 30))
            pen.setWidthF(0.9)
            pen.setCosmetic(True)
            if primitive["partial"]:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawEllipse(bounds)
            return
        start_angle = 90.0
        visible_count = sum(component["visible"] for component in components)
        for component in components:
            span_angle = 360.0 * component["fraction"]
            if not component["visible"] or span_angle <= 1e-8:
                start_angle += span_angle
                continue
            fill = QColor(component["colour"])
            gradient = QRadialGradient(
                QPointF(point.x() - 0.28 * radius, point.y() - 0.28 * radius),
                1.35 * radius,
                QPointF(point.x() - 0.30 * radius, point.y() - 0.30 * radius),
            )
            highlight = fill.lighter(175)
            middle = QColor(fill)
            shadow = fill.darker(135)
            gradient.setColorAt(0.0, QColor(255, 255, 255))
            gradient.setColorAt(0.25, highlight)
            gradient.setColorAt(0.72, middle)
            gradient.setColorAt(1.0, shadow)
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.PenStyle.NoPen)
            if visible_count == 1 and span_angle >= 359.999:
                painter.drawEllipse(bounds)
            else:
                painter.drawPie(
                    bounds,
                    int(round(start_angle * 16.0)),
                    int(round(span_angle * 16.0)),
                )
            start_angle += span_angle
        pen = QPen(QColor(primitive["outline"]).darker(150))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(bounds)

    def _draw_bond(self, painter: QPainter, primitive: dict) -> None:
        first = primitive["first"]
        second = primitive["second"]
        if self.engraving:
            pen = QPen(QColor("#777777"))
            pen.setWidthF(1.8)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(first, second)
            return
        midpoint = QPointF(
            0.5 * (first.x() + second.x()),
            0.5 * (first.y() + second.y()),
        )
        for start, finish, colour in (
            (first, midpoint, primitive["first_colour"]),
            (midpoint, second, primitive["second_colour"]),
        ):
            pen = QPen(QColor(colour))
            pen.setWidthF(3.2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(start, finish)

    def _draw_cell(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_cell:
            return
        vertices = self.scene.cell_vertices @ self.orientation.T
        pen = QPen(QColor(75, 75, 75))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for first, second in self.scene.cell_edges:
            painter.drawLine(self._project(vertices[first])[0], self._project(vertices[second])[0])

    def _draw_basis(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_basis:
            return
        directions = self.orientation @ (
            self.scene.basis_vectors
            / np.linalg.norm(self.scene.basis_vectors, axis=0)
        )
        origin = QPointF(56.0, self.height() - 52.0)
        colours = (QColor("#db2828"), QColor("#25a43a"), QColor("#2155dc"))
        for index, (name, colour) in enumerate(zip("abc", colours)):
            x, y = float(directions[0, index]), float(directions[1, index])
            length = math.hypot(x, y)
            pen = QPen(colour)
            pen.setWidthF(2.4)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(colour))
            if length <= 0.04:
                painter.drawEllipse(QRectF(origin.x() - 3, origin.y() - 3, 6, 6))
                painter.drawText(QPointF(origin.x() + 7, origin.y() - 7), name)
                continue
            target = QPointF(origin.x() + 43.0 * x, origin.y() - 43.0 * y)
            painter.drawLine(origin, target)
            vector = np.array([target.x() - origin.x(), target.y() - origin.y()])
            vector /= np.linalg.norm(vector)
            normal = np.array([-vector[1], vector[0]])
            tip = np.array([target.x(), target.y()])
            left = tip - 9.0 * vector + 4.0 * normal
            right = tip - 9.0 * vector - 4.0 * normal
            painter.drawPolygon(
                QPolygonF(
                    [target, QPointF(*left), QPointF(*right)]
                )
            )
            painter.drawText(
                QPointF(target.x() + 7.0 * x, target.y() - 7.0 * y), name
            )
        painter.setBrush(QBrush(QColor("#777777")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(origin.x() - 2.5, origin.y() - 2.5, 5, 5))

    def _draw_legend(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_atoms:
            return
        x, y = 12.0, 18.0
        metrics = QFontMetrics(painter.font())
        visible_elements = {
            component.element
            for atom in self.scene.atoms
            for component in atom.components
            if self._component_visible(component)
        }
        for element in self.scene.elements:
            if element not in visible_elements:
                continue
            component = next(
                component
                for atom in self.scene.atoms
                for component in atom.components
                if component.element == element
            )
            colour = (
                _engraving_colour(element, component.colour)
                if self.engraving
                else self._component_colour(component)
            )
            painter.setPen(QPen(colour.darker(150)))
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(QRectF(x, y - 8.0, 10.0, 10.0))
            painter.setPen(QPen(self._foreground_colour()))
            painter.drawText(QPointF(x + 15.0, y), element)
            x += 22.0 + metrics.horizontalAdvance(element)
            if x > self.width() - 80.0:
                x = 12.0
                y += metrics.height() + 4.0

    def _foreground_colour(self) -> QColor:
        return QColor(25, 25, 25) if self.engraving else self.palette().text().color()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            background = QColor(250, 250, 247) if self.engraving else self.palette().base().color()
            painter.fillRect(self.rect(), background)
            if self.scene is None:
                painter.setPen(QPen(self._foreground_colour()))
                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    self.empty_text,
                )
                return
            for primitive in self._primitives():
                if primitive["kind"] == "face":
                    self._draw_face(painter, primitive)
                elif primitive["kind"] == "bond":
                    self._draw_bond(painter, primitive)
                else:
                    self._draw_atom(painter, primitive)
            self._draw_cell(painter)
            self._draw_basis(painter)
            self._draw_legend(painter)
            painter.setPen(QPen(self._foreground_colour()))
            if self.title_text:
                painter.drawText(
                    QRectF(8.0, 7.0, self.width() - 16.0, 24.0),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    self.title_text,
                )
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:
        if (
            event.button()
            in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
            and self.scene is not None
        ):
            self._last_mouse = event.position()
            self._drag_button = event.button()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        dx = position.x() - self._last_mouse.x()
        dy = position.y() - self._last_mouse.y()
        self._last_mouse = position
        if self._drag_button == Qt.MouseButton.LeftButton:
            if self.rotation_enabled:
                self.orientation = screen_drag_orientation(
                    self.orientation, float(dx), float(dy)
                )
                self.orientation_changed.emit(self.orientation.copy())
                self.update()
        elif self._drag_button == Qt.MouseButton.RightButton:
            self.pan_x += float(dx)
            self.pan_y += float(dy)
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == self._drag_button and self._last_mouse is not None:
            rotated = self._drag_button == Qt.MouseButton.LeftButton
            self._last_mouse = None
            self._drag_button = None
            if rotated and self.rotation_enabled:
                self.interaction_finished.emit()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if self.scene is None:
            super().wheelEvent(event)
            return
        self.zoom_by(event.angleDelta().y() / 120.0)
        event.accept()


CrystalCanvas = UnitCellViewer


__all__ = ["CrystalCanvas", "UnitCellViewer", "screen_drag_orientation"]
