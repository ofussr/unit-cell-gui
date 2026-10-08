"""Atom data compatibility and real OpenGL buffer/lifecycle regressions."""

import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
import shiboken6
from unit_cell_gui import Atom, AtomComponent, Scene, StyleOverrides, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from unit_cell_gui.gl_geometry import indexed_sphere_mesh, sphere_mesh
from test_unit_cell_gui import rotation, sample_scene


def atom_scene():
    prototype = sample_scene()
    components = tuple(
        AtomComponent(f"c{i}", f"C{i}", "C", .1, f"#{i * 25:02x}80a0", .15 + i * .03)
        for i in range(8)
    )
    return Scene(
        (Atom("mixed", "Mixed", components),
         Atom("external", "External", (components[0],), external=True)),
        [[-.3, .2, .1], [.8, -.4, .2]], [],
        prototype.cell_vertices, prototype.cell_edges, prototype.basis_vectors,
        base_radius=2.,
    )


class AtomInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.set_scene(atom_scene())

    def tearDown(self):
        self.viewer.close()

    def test_indices_reproduce_original_triangle_positions_and_normals(self):
        positions, normals, indices = indexed_sphere_mesh()
        original_positions, original_normals = sphere_mesh(np.zeros(3), 1.)
        np.testing.assert_array_equal(positions[indices], original_positions)
        np.testing.assert_array_equal(normals[indices], original_normals)
        self.assertLess(len(positions), len(original_positions) / 4)
        self.assertEqual(indices.dtype, np.uint32)

    def test_instances_preserve_coordinates_radius_eight_sectors_and_partial_occupancy(self):
        data = self.viewer._build_atom_instances()
        self.assertEqual(data.shape, (2, 36))
        self.assertEqual(data.nbytes, 2 * 144)
        self.assertTrue(data.flags.c_contiguous)
        np.testing.assert_allclose(data[:, :3], self.viewer.scene.atom_centers)
        # Weighted radius of the eight components; external atom has its own radius.
        np.testing.assert_allclose(data[:, 3], [.255, .15])
        np.testing.assert_allclose(data[0, 28:], np.arange(1, 9) * .1)
        np.testing.assert_allclose(data[1, 28:], .1)
        np.testing.assert_allclose(data[0, 4:28].reshape(8, 3),
                                   [[i * 25 / 255, 128 / 255, 160 / 255] for i in range(8)], atol=1e-6)

    def test_visibility_colours_scale_and_engraving_keep_instance_order_and_materials(self):
        self.viewer.set_display_options(show_external_atoms=False, atom_scale=2.)
        self.viewer.set_style_overrides(StyleOverrides(
            atom_component_visibility={"c1": False}, atom_component_colours={"c7": "#ff0000"},
        ))
        data = self.viewer._build_atom_instances()
        self.assertEqual(len(data), 1)
        self.assertAlmostEqual(float(data[0, 3]), .51, places=6)
        np.testing.assert_array_equal(data[0, 7:10], [1, 1, 1])
        np.testing.assert_array_equal(data[0, 25:28], [1, 0, 0])
        self.viewer.set_display_options(engraving=True)
        engraved = self.viewer._build_atom_instances()
        np.testing.assert_array_equal(engraved[0, 28:], np.ones(8))
        self.assertEqual(engraved[0, 4], engraved[0, 5])
        self.assertEqual(engraved[0, 5], engraved[0, 6])
        self.viewer.set_style_overrides(StyleOverrides(atom_component_visibility={f"c{i}": False for i in range(8)}))
        self.assertEqual(self.viewer._build_atom_instances().shape, (0, 36))
        self.viewer.set_scene(None)
        self.assertEqual(self.viewer._build_atom_instances().shape, (0, 36))

    def test_camera_and_show_atoms_do_not_invalidate_instances(self):
        self.viewer._atom_dirty = False
        self.viewer.set_orientation(rotation(1, 35))
        self.viewer.zoom_by(1)
        self.viewer.set_display_options(show_atoms=False)
        self.assertFalse(self.viewer._atom_dirty)
        self.viewer.set_display_options(show_atoms=True)
        self.assertFalse(self.viewer._atom_dirty)


class OpenGLAtomInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() in {"offscreen", "minimal"}:
            raise unittest.SkipTest("A window system with OpenGL is required")

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.resize(640, 480)
        self.viewer.set_scene(atom_scene())
        self.viewer.set_display_options(engraving=True)
        self.viewer.show()
        self.app.processEvents()
        if not self.viewer.isValid():
            self.viewer.close()
            self.skipTest("No OpenGL context could be created")
        self.assertIsNone(self.viewer.opengl_error)

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)
        self.app.processEvents()

    def pixels(self):
        image = self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.viewer.opengl_error)
        self.assertFalse(image.isNull())
        return np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes()).copy()

    def test_instance_divisors_buffer_sizes_and_index_binding(self):
        from OpenGL.GL import (
            GL_ARRAY_BUFFER, GL_BUFFER_SIZE, GL_ELEMENT_ARRAY_BUFFER,
            GL_ELEMENT_ARRAY_BUFFER_BINDING, GL_VERTEX_ATTRIB_ARRAY_DIVISOR,
            glBindBuffer, glBindVertexArray, glGetBufferParameteriv,
            glGetIntegerv, glGetVertexAttribiv,
        )
        self.pixels()
        self.viewer.makeCurrent()
        try:
            glBindVertexArray(self.viewer._atom_vao)
            self.assertEqual(int(glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING)), self.viewer._atom_ebo)
            for location in range(13):
                divisor = np.asarray(glGetVertexAttribiv(location, GL_VERTEX_ATTRIB_ARRAY_DIVISOR)).ravel()[0]
                self.assertEqual(divisor, 0 if location < 2 else 1)
            glBindBuffer(GL_ARRAY_BUFFER, self.viewer._atom_vbo)
            self.assertEqual(int(glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)), self.viewer._atom_count * 144)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.viewer._atom_ebo)
            self.assertEqual(int(glGetBufferParameteriv(GL_ELEMENT_ARRAY_BUFFER, GL_BUFFER_SIZE)), 2160 * 4)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        finally:
            self.viewer.doneCurrent()

    def test_indexed_draw_and_material_changes_restore_frame_without_mesh_upload(self):
        before = self.pixels()
        mesh = self.viewer._atom_mesh_vbo
        with patch.object(renderer, "glDrawElementsInstanced", wraps=renderer.glDrawElementsInstanced) as draw, \
             patch.object(self.viewer, "_create_atom_buffer", wraps=self.viewer._create_atom_buffer) as create:
            self.viewer.set_display_options(atom_scale=1.4, engraving=False)
            self.assertFalse(np.array_equal(self.pixels(), before))
            self.viewer.set_display_options(atom_scale=1., engraving=True)
            np.testing.assert_array_equal(self.pixels(), before)
            create.assert_not_called()
            self.assertEqual(self.viewer._atom_mesh_vbo, mesh)
            self.assertTrue(draw.called)
            self.assertEqual(draw.call_args.args[1], 2160)
            self.assertEqual(draw.call_args.args[4], self.viewer._atom_count)

    def test_atom_buffers_cleanup_reinitialization_and_empty_scene(self):
        from OpenGL.GL import glIsBuffer
        before = self.pixels()
        buffers = [self.viewer._atom_vbo, self.viewer._atom_mesh_vbo, self.viewer._atom_ebo]
        self.viewer.cleanup()
        self.viewer.makeCurrent()
        try:
            self.assertTrue(all(not glIsBuffer(buffer) for buffer in buffers))
            self.assertEqual(self.viewer._atom_index_count, 0)
            self.viewer.initializeGL()
        finally:
            self.viewer.doneCurrent()
        np.testing.assert_array_equal(self.pixels(), before)
        self.viewer.set_scene(None)
        self.pixels()
        with patch.object(renderer, "glDrawElementsInstanced", wraps=renderer.glDrawElementsInstanced) as draw:
            self.pixels()
            draw.assert_not_called()
        self.viewer.set_scene(atom_scene())
        self.pixels()
        self.assertEqual(self.viewer._atom_count, 2)
        self.assertIsNone(self.viewer.opengl_error)
