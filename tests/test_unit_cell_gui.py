from pathlib import Path
import os
import subprocess
import sys
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from unit_cell_gui import (
    Atom, AtomComponent, CrystalCanvas, DisplayOptions, OpenGLUnitCellViewer, Polyhedron, Scene, StyleOverrides, UnitCellViewer,
    face_hatch_segments, hatch_division_count, occupancy_fractions,
    propagated_hatch_directions, quadrilateral_hatch_line_count,
)
from unit_cell_gui.viewer import _polyhedron_hatches


def rotation(axis, degrees):
    angle = np.radians(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.eye(3)
    first, second = ((1, 2), (0, 2), (0, 1))[axis]
    result[first, first] = result[second, second] = cosine
    result[first, second], result[second, first] = -sine, sine
    return result


def on_perimeter(point, polygon, tolerance=1e-8):
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        fraction = np.dot(point - first, edge) / np.dot(edge, edge)
        if -tolerance <= fraction <= 1 + tolerance:
            if np.linalg.norm(point - first - fraction * edge) < tolerance:
                return True
    return False


def sample_scene():
    niobium = AtomComponent("Nb:Nb1", "Nb1", "Nb", .4, "#55aadd", .35)
    tantalum = AtomComponent("Ta:Ta1", "Ta1", "Ta", .6, "#999999", .35)
    oxygen = AtomComponent("O:O1", "O1", "O", 1, "#ee4444", .25)
    atoms = (
        Atom("centre", "Nb1/Ta1", (niobium, tantalum)),
        Atom("oxygen", "O1", (oxygen,)),
    )
    vertices = np.array([
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ], float)
    faces = (
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    )
    polyhedron = Polyhedron(np.zeros(3), vertices, faces, "centre", "Nb1/Ta1",
                            (niobium, tantalum))
    cell = np.array([(x, y, z) for x in (-2, 2) for y in (-2, 2) for z in (-2, 2)])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if np.count_nonzero(cell[i] != cell[j]) == 1]
    return Scene(atoms, np.array([(0, 0, 0), (1, 0, 0)]), np.array([(0, 1)]),
                 cell, np.array(edges), np.eye(3), (polyhedron,), 4.0)


class PureApiTests(unittest.TestCase):
    def test_scene_validates_ready_geometry(self):
        scene = sample_scene()
        self.assertEqual(scene.elements, ("Nb", "O", "Ta"))
        self.assertEqual(scene.polyhedra[0].coordination_number, 6)
        with self.assertRaises(ValueError):
            Scene(scene.atoms, np.zeros((1, 3)), scene.bonds, scene.cell_vertices,
                  scene.cell_edges, scene.basis_vectors)
        with self.assertRaises(ValueError):
            Scene(scene.atoms, scene.atom_centers, scene.bonds, scene.cell_vertices,
                  scene.cell_edges, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            DisplayOptions(atom_scale=0)

    def test_hatching_and_occupancy_are_part_of_the_package(self):
        components = sample_scene().atoms[0].components
        self.assertEqual(occupancy_fractions(components), (.4, .6))
        self.assertEqual(quadrilateral_hatch_line_count(4), 9)
        self.assertEqual(hatch_division_count(1, 22, 50), 17)
        square = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float)
        direction = propagated_hatch_directions(((0, 1, 2, 3),), square, (0,))[0]
        self.assertEqual(len(face_hatch_segments(square, direction, 9)), 9)

    def test_package_source_has_no_xrd_workbench_dependency(self):
        root = Path(__file__).resolve().parents[1] / "src" / "unit_cell_gui"
        source = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))
        self.assertNotIn("xrd_workbench", source)

    def test_top_level_package_import_is_lazy_about_qt(self):
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys, unit_cell_gui; "
                "raise SystemExit(any(name.startswith('PySide6') for name in sys.modules))"
            )],
            check=False,
        )
        self.assertEqual(result.returncode, 0)


class HatchingRegressionTests(unittest.TestCase):
    def test_quadrilateral_remains_one_face_with_full_parallel_lines(self):
        square = np.array([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], float)
        points = square @ (rotation(1, 28) @ rotation(0, 17)).T
        direction = propagated_hatch_directions(((0, 1, 2, 3),), points, (0,))[0]
        apex = int(np.argmax(points[:, 2]))
        diagonal = points[(apex + 1) % 4] - points[(apex - 1) % 4]
        self.assertLess(np.linalg.norm(np.cross(direction, diagonal)), 1e-10)
        segments = face_hatch_segments(points, direction, 9)
        self.assertEqual(len(segments), 9)
        for first, second in segments:
            self.assertTrue(on_perimeter(first, points))
            self.assertTrue(on_perimeter(second, points))
            self.assertLess(np.linalg.norm(np.cross(second - first, direction)), 1e-9)

    def test_far_face_direction_is_refolded_through_each_shared_edge(self):
        faces = ((0, 1, 2), (1, 3, 2), (2, 3, 4))
        vertices = np.array([
            (0, 1, 3), (-1, 0, 2), (1, 0, 2), (0, -1, 0), (-.3, -2, -3)
        ], float)
        directions = propagated_hatch_directions(faces, vertices, (0, 1, 2))
        for index, face in enumerate(faces):
            points = vertices[list(face)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normal /= np.linalg.norm(normal)
            self.assertAlmostEqual(float(directions[index] @ normal), 0.0)
        for parent, child in ((0, 1), (1, 2)):
            first, second = sorted(set(faces[parent]).intersection(faces[child]))
            edge = vertices[second] - vertices[first]
            edge /= np.linalg.norm(edge)
            self.assertAlmostEqual(
                float(directions[parent] @ edge),
                float(directions[child] @ edge),
            )

    def test_actual_cube_plan_draws_no_virtual_diagonals_or_subfaces(self):
        component = AtomComponent("X:1", "X1", "X", 1, "#777777", .3)
        vertices = np.array([
            (x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
        ], float)
        faces = (
            (3, 2, 0, 1), (5, 1, 0, 4), (6, 4, 0, 2),
            (7, 3, 1, 5), (7, 6, 2, 3), (7, 5, 4, 6),
        )
        polyhedron = Polyhedron(np.zeros(3), vertices, faces, "X:1", "X1", (component,))
        camera = vertices @ (rotation(2, 9) @ rotation(1, 31) @ rotation(0, 17)).T
        projected = [QPointF(float(point[0]), -float(point[1])) for point in camera]
        normals = {}
        for index, face in enumerate(faces):
            points = camera[list(face)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            normals[index] = normal / np.linalg.norm(normal)
        visible = [index for index, normal in normals.items() if normal[2] > 1e-9]
        hatches = _polyhedron_hatches(
            polyhedron, camera, projected, visible, normals, 4,
            lambda point: QPointF(float(point[0]), -float(point[1])),
        )
        for index in visible:
            lines = hatches[index]
            expected = 0 if normals[index][2] >= .999 else 9
            self.assertEqual(len(lines), expected)
            polygon = np.array([
                (projected[vertex].x(), projected[vertex].y()) for vertex in faces[index]
            ])
            vectors = []
            for first, second in lines:
                a = np.array((first.x(), first.y()))
                b = np.array((second.x(), second.y()))
                self.assertTrue(on_perimeter(a, polygon))
                self.assertTrue(on_perimeter(b, polygon))
                vectors.append(b - a)
            if vectors:
                reference = vectors[0] / np.linalg.norm(vectors[0])
                self.assertTrue(all(abs(vector[0] * reference[1] - vector[1] * reference[0]) < 1e-8
                                    for vector in vectors))


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_legacy_viewer_names_resolve_to_opengl_backend(self):
        self.assertIs(UnitCellViewer, OpenGLUnitCellViewer)
        self.assertIs(CrystalCanvas, UnitCellViewer)

    def test_input_output_orientation_and_camera_controls(self):
        viewer = UnitCellViewer()
        viewer.resize(500, 400)
        scene = sample_scene()
        viewer.set_scene(scene)
        viewer.set_display_options(DisplayOptions(show_polyhedra=True))
        viewer.show()
        self.app.processEvents()
        received = []
        finished = []
        viewer.orientation_changed.connect(received.append)
        viewer.interaction_finished.connect(lambda: finished.append(True))
        before = viewer.orientation.copy()
        start = QPoint(250, 200)
        QTest.mousePress(viewer, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer, start + QPoint(20, 10))
        QTest.mouseRelease(viewer, Qt.MouseButton.LeftButton, pos=start + QPoint(20, 10))
        self.app.processEvents()
        self.assertTrue(received)
        self.assertEqual(finished, [True])
        self.assertFalse(np.allclose(viewer.orientation, before))
        requested = np.diag([-1., -1., 1.])
        viewer.set_orientation(requested)
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(viewer.orientation, requested)
        viewer.set_display_options(rotation_enabled=False)
        QTest.mousePress(viewer, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewer, start + QPoint(30, 15))
        QTest.mouseRelease(viewer, Qt.MouseButton.LeftButton, pos=start + QPoint(30, 15))
        np.testing.assert_array_equal(viewer.orientation, requested)
        self.assertEqual(len(received), 1)
        viewer.close()

    def test_rendering_does_not_mutate_input_scene(self):
        viewer = UnitCellViewer()
        scene = sample_scene()
        vertices = scene.polyhedra[0].vertices.copy()
        centers = scene.atom_centers.copy()
        viewer.set_scene(scene)
        viewer.resize(500, 400)
        image = viewer.grab().toImage()
        self.assertFalse(image.isNull())
        np.testing.assert_array_equal(scene.polyhedra[0].vertices, vertices)
        np.testing.assert_array_equal(scene.atom_centers, centers)

    def test_style_overrides_are_copied_at_the_public_boundary(self):
        viewer = UnitCellViewer()
        visibility = {"Nb:Nb1": False}
        viewer.set_style_overrides(StyleOverrides(
            atom_component_visibility=visibility,
            polyhedron_opaque_sites={"centre"},
        ))
        visibility["Nb:Nb1"] = True
        self.assertFalse(viewer.atom_component_visibility["Nb:Nb1"])
        self.assertEqual(viewer.polyhedron_opaque_sites, {"centre"})


if __name__ == "__main__":
    unittest.main()
