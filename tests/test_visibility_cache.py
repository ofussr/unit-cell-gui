"""Visibility, finite-face clipping and view-dependent cache regressions."""

from contextlib import ExitStack
import os
from types import SimpleNamespace
import unittest
import shiboken6
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from unit_cell_gui import Atom, AtomComponent, DisplayOptions, Polyhedron, Scene, UnitCellViewer, StyleOverrides
from unit_cell_gui import opengl_viewer as renderer
from test_unit_cell_gui import rotation, sample_scene


def repeated_scene(count=5):
    original = sample_scene()
    prototype = original.polyhedra[0]
    polyhedra, atoms, centers = [], [], []
    for index in range(count):
        center = np.array([0.7 * index, 0.0, float(index - count // 2)])
        key = f"p{index}"
        components = (AtomComponent(f"Nb:{index}", key, "Nb", 1, "#55aadd", .35),)
        polyhedra.append(Polyhedron(center, prototype.vertices + center, prototype.faces, key, key, components))
        atoms.append(Atom(key, key, components))
        centers.append(center)
        # This sphere intersects the finite polygons near a ligand vertex.
        atoms.append(Atom(f"o{index}", f"O{index}", original.atoms[1].components, external=index % 2 == 1))
        centers.append(center + np.array([1., 0., 0.]))
    return Scene(atoms, centers, np.empty((0, 2), int), original.cell_vertices,
                 original.cell_edges, original.basis_vectors, polyhedra, 6.)


class VisibilityCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.resize(640, 480)
        self.viewer.set_scene(repeated_scene())
        self.viewer.set_display_options(DisplayOptions(show_polyhedra=True))

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)

    def test_site_visibility_keeps_static_geometry_and_intersections(self):
        before = self.viewer._build_static_geometry()
        intersections, ranges = self.viewer._build_atom_face_intersection_geometry()
        self.assertGreater(len(intersections), 0)
        self.assertEqual({item.site_key for item in ranges}, {f"p{i}" for i in range(5)})
        self.viewer._gpu_dirty = self.viewer._intersection_dirty = False
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={"p2": False}))
        self.assertFalse(self.viewer._gpu_dirty)
        self.assertFalse(self.viewer._intersection_dirty)
        after = self.viewer._build_static_geometry()
        for index in (1, 3, 5):
            np.testing.assert_array_equal(before[index], after[index])
        visible = self.viewer._visible_polyhedron_ranges(ranges)
        self.assertNotIn("p2", {item.site_key for item in visible})
        self.assertEqual(sum(item.count for item in ranges), len(intersections))
        self.viewer.set_style_overrides(StyleOverrides())
        self.assertEqual(sum(item.count for item in self.viewer._visible_polyhedron_ranges(ranges)), len(intersections))

    def test_draw_calls_filter_opaque_faces_transparent_faces_edges_and_sections(self):
        self.viewer.set_style_overrides(StyleOverrides(
            polyhedron_site_visibility={"p1": False, "p2": False},
            polyhedron_opaque_sites={"p0", "p1"},
        ))
        with patch.object(self.viewer, "_upload_static", side_effect=lambda _vao, _vbo, data: len(data)):
            self.viewer._rebuild_gpu_cache()
            self.viewer._rebuild_intersection_cache()
        with ExitStack() as stack:
            for name in ("glBindVertexArray", "glEnable", "glBlendFunc", "glDepthMask", "glDisable", "glUseProgram"):
                stack.enter_context(patch.object(renderer, name))
            stack.enter_context(patch.object(self.viewer, "_set_edge_uniforms"))
            stack.enter_context(patch.object(self.viewer, "_set_instance_uniforms"))
            stack.enter_context(patch.object(self.viewer, "_bind_primitive"))
            draw_ordered = stack.enter_context(patch.object(self.viewer, "_draw_ordered_edge_instances"))
            draw_instances = stack.enter_context(patch.object(self.viewer, "_draw_primitive_instances"))
            draw = stack.enter_context(patch.object(renderer, "glDrawArrays"))
            for method, ranges in (
                (self.viewer._draw_opaque_polyhedra, self.viewer._opaque_ranges),
                (self.viewer._draw_sorted_transparency, self.viewer._face_ranges+self.viewer._face_edge_ranges),
                (self.viewer._draw_face_edges, self.viewer._face_edge_ranges),
                (self.viewer._draw_atom_face_intersections, self.viewer._intersection_ranges),
            ):
                draw.reset_mock(); draw_instances.reset_mock(); draw_ordered.reset_mock()
                method()
                active = draw if method == self.viewer._draw_atom_face_intersections else draw_instances
                drawn = sorted((call.args[1], call.args[2]) for call in active.call_args_list)
                if method == self.viewer._draw_face_edges:
                    drawn = sorted((item.first, item.count) for call in draw_ordered.call_args_list for item in call.args[0])
                expected = sorted((item.first, item.count) for item in ranges if item.site_key not in {"p1", "p2"})
                if method == self.viewer._draw_sorted_transparency:
                    drawn_indices=sorted((call.args[0],i) for call in draw_instances.call_args_list
                                         for i in range(call.args[1],call.args[1]+call.args[2]))
                    expected_indices=sorted((renderer.GL_TRIANGLES if item.vertices.shape[0]>2 else renderer.GL_LINES,i)
                                            for item in ranges if item.site_key not in {"p1","p2"}
                                            for i in range(item.first,item.first+item.count))
                    self.assertEqual(drawn_indices,expected_indices)
                else:
                    self.assertEqual(drawn, expected)
                self.viewer.show_polyhedra = False
                draw.reset_mock(); draw_instances.reset_mock(); draw_ordered.reset_mock()
                method()
                draw.assert_not_called(); draw_instances.assert_not_called(); draw_ordered.assert_not_called()
                self.viewer.show_polyhedra = True

    def _fake_gpu(self, stack):
        stack.enter_context(patch.object(self.viewer, "context", return_value=SimpleNamespace(isValid=lambda: True)))
        stack.enter_context(patch.object(self.viewer, "_create_buffer", side_effect=[(100+i, 200+i) for i in range(20)]))
        return stack.enter_context(patch.object(self.viewer, "_upload_static", side_effect=lambda _vao, _vbo, data: len(data)))

    def test_hiding_and_reshowing_middle_polyhedron_reuses_hatch_gpu_buffers(self):
        self.viewer.set_display_options(engraving=True)
        self.viewer.set_orientation(rotation(0, 17))
        with ExitStack() as stack:
            upload = self._fake_gpu(stack)
            build = stack.enter_context(patch.object(self.viewer, "_build_polyhedron_hatch_geometry", wraps=self.viewer._build_polyhedron_hatch_geometry))
            self.viewer._rebuild_hatch_cache()
            self.assertEqual(upload.call_count, 5)
            self.assertGreater(self.viewer._hatch_count, 0)
            upload.reset_mock(); build.reset_mock()
            self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={"p2": False}))
            self.viewer._rebuild_hatch_cache()
            self.assertNotIn(2, self.viewer._hatch_draw_indices)
            upload.assert_not_called(); build.assert_not_called()
            self.viewer.set_style_overrides(StyleOverrides())
            self.viewer._rebuild_hatch_cache()
            self.assertIn(2, self.viewer._hatch_draw_indices)
            upload.assert_not_called(); build.assert_not_called()

    def test_depth_extreme_changes_only_affected_hatch_variants_and_reuses_cpu_cache(self):
        self.viewer.set_display_options(engraving=True)
        with ExitStack() as stack:
            upload = self._fake_gpu(stack)
            build = stack.enter_context(patch.object(self.viewer, "_build_polyhedron_hatch_geometry", wraps=self.viewer._build_polyhedron_hatch_geometry))
            original = dict(self.viewer._hatch_plan())
            self.viewer._rebuild_hatch_cache()
            upload.reset_mock(); build.reset_mock()
            self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={"p4": False}))
            updated = dict(self.viewer._hatch_plan())
            affected = {index for index, divisions in updated.items() if divisions != original[index]}
            self.assertTrue(affected)
            self.viewer._rebuild_hatch_cache()
            self.assertEqual({call.args[0] for call in build.call_args_list}, affected)
            self.assertEqual(upload.call_count, len(affected))
            build.reset_mock()
            self.viewer.set_style_overrides(StyleOverrides())
            self.viewer._rebuild_hatch_cache()
            build.assert_not_called()

    def test_hatch_cache_updates_with_view_and_is_bounded(self):
        self.viewer.set_display_options(engraving=True)
        first = self.viewer._build_hatch_geometry()
        self.viewer.set_orientation(rotation(0, 64) @ rotation(1, 33))
        second = self.viewer._build_hatch_geometry()
        self.assertTrue(first.shape != second.shape or not np.array_equal(first, second))
        with patch.object(self.viewer, "_build_polyhedron_hatch_geometry", wraps=self.viewer._build_polyhedron_hatch_geometry) as build:
            self.viewer.zoom_by(1)
            self.viewer._build_hatch_geometry()
            self.assertEqual(build.call_count, 5)
        for divisions in range(1, 12):
            self.viewer._cached_polyhedron_hatches(0, divisions)
        self.assertLessEqual(len(self.viewer._hatch_mesh_cache[0]), renderer._HATCH_VARIANTS_PER_POLYHEDRON)
        self.viewer.set_scene(repeated_scene(1))
        self.assertFalse(self.viewer._hatch_mesh_cache)
        self.assertIsNone(self.viewer._hatch_view_key)

    def test_atom_size_and_visibility_still_invalidate_intersections(self):
        self.viewer._intersection_dirty = False
        self.viewer.set_display_options(atom_scale=1.3)
        self.assertTrue(self.viewer._intersection_dirty)
        self.viewer._intersection_dirty = False
        self.viewer.set_style_overrides(StyleOverrides(atom_component_visibility={"Nb:0": False}))
        self.assertTrue(self.viewer._intersection_dirty)

    def test_one_site_selection_hides_all_periodic_images_of_that_site(self):
        scene = self.viewer.scene
        polyhedra = [Polyhedron(poly.center, poly.vertices, poly.faces,
                                "shared" if index in (1, 3) else poly.site_key,
                                poly.site_label, poly.components)
                     for index, poly in enumerate(scene.polyhedra)]
        self.viewer.set_scene(Scene(scene.atoms, scene.atom_centers, scene.bonds,
                                   scene.cell_vertices, scene.cell_edges, scene.basis_vectors,
                                   polyhedra, scene.base_radius))
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={"shared": False}))
        geometry = self.viewer._build_static_geometry()
        visible = self.viewer._visible_polyhedron_ranges(geometry[4])
        self.assertEqual({item.site_key for item in visible}, {"p0", "p2", "p4"})
        self.assertEqual(len(visible), 3 * 8)

    def test_empty_scene_and_all_hidden_polyhedra(self):
        self.viewer.set_display_options(engraving=True)
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={f"p{i}": False for i in range(5)}))
        self.assertEqual(self.viewer._hatch_plan(), [])
        self.assertEqual(len(self.viewer._build_hatch_geometry()), 0)
        self.viewer.set_scene(None)
        geometry, ranges = self.viewer._build_atom_face_intersection_geometry()
        self.assertEqual(len(geometry), 0)
        self.assertEqual(ranges, [])
        self.assertEqual(len(self.viewer._build_hatch_geometry()), 0)

    def test_vectorized_finite_face_test_matches_scalar_on_boundaries_and_both_windings(self):
        rng = np.random.default_rng(221)
        for scale in (.1, 1., 12.):
            polygon = scale * np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
            points = np.vstack((rng.uniform(-2*scale, 2*scale, (200, 3)), polygon,
                                [[scale, 0., 0.], [scale + 1.e-10, 0., 0.], [scale + 1.e-6, 0., 0.]]))
            for poly in (polygon, polygon[::-1]):
                normal = np.array([0., 0., 1.])
                expected = [self.viewer._point_in_convex_face(point, poly, normal) for point in points]
                np.testing.assert_array_equal(self.viewer._points_in_convex_face(points, poly, normal), expected)


class OpenGLVisibilityTests(unittest.TestCase):
    """Run with a window system; offscreen Qt cannot validate framebuffer output."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() in {"offscreen", "minimal"}:
            raise unittest.SkipTest("A window system with OpenGL is required")

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.resize(640, 480)
        self.viewer.set_scene(repeated_scene())
        self.viewer.set_display_options(DisplayOptions(show_polyhedra=True, engraving=True))
        self.viewer.set_orientation(rotation(0, 17) @ rotation(1, 21))
        self.viewer.show()
        self.app.processEvents()
        if not self.viewer.isValid():
            self.viewer.close()
            shiboken6.delete(self.viewer)
            self.skipTest("No OpenGL context could be created")
        self.assertIsNone(self.viewer.opengl_error)

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)

    def pixels(self):
        image = self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertFalse(image.isNull())
        return np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes()).copy()

    def test_site_toggle_changes_frame_and_restores_it_without_static_rebuilds(self):
        before = self.pixels()
        with patch.object(self.viewer, "_rebuild_gpu_cache", wraps=self.viewer._rebuild_gpu_cache) as static, \
             patch.object(self.viewer, "_rebuild_intersection_cache", wraps=self.viewer._rebuild_intersection_cache) as sections:
            self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={"p2": False}))
            hidden = self.pixels()
            self.assertFalse(np.array_equal(before, hidden))
            self.viewer.set_style_overrides(StyleOverrides())
            np.testing.assert_array_equal(self.pixels(), before)
            static.assert_not_called()
            sections.assert_not_called()

    def test_scene_clear_releases_per_polyhedron_gpu_resources(self):
        self.pixels()
        from OpenGL.GL import glIsBuffer
        buffers = [buffer.vbo for buffer in self.viewer._hatch_buffers.values()]
        self.assertTrue(buffers)
        self.viewer.set_scene(None)
        self.pixels()
        self.assertFalse(self.viewer._hatch_buffers)
        self.viewer.makeCurrent()
        try:
            self.assertTrue(all(not glIsBuffer(buffer) for buffer in buffers))
        finally:
            self.viewer.doneCurrent()

    def test_gpu_resources_rebuild_after_context_cleanup(self):
        before = self.pixels()
        self.viewer.cleanup()
        self.assertFalse(self.viewer._hatch_buffers)
        self.viewer.makeCurrent()
        self.viewer.initializeGL()
        self.viewer.doneCurrent()
        np.testing.assert_array_equal(self.pixels(), before)
        self.assertTrue(self.viewer._hatch_buffers)


if __name__ == "__main__":
    unittest.main()
