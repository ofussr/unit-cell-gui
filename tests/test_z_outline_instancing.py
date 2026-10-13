"""Instanced silhouette fidelity, camera caching and GL resource lifetime."""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QMouseEvent
import shiboken6
from unit_cell_gui import Atom, AtomComponent, Scene, StyleOverrides, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from test_unit_cell_gui import rotation, sample_scene


def outline_scene():
    base = sample_scene()
    full = AtomComponent('full', 'Full', 'C', 1., '#5588bb', .3)
    partial = AtomComponent('partial', 'Partial', 'O', .4, '#bb8855', .4)
    return Scene((Atom('a', 'A', (full,)), Atom('b', 'B', (partial,)),
                  Atom('ext', 'External', (full,), external=True)),
                 [[-.3, 0, .1], [.2, 0, .4], [.7, .1, -.1]], [],
                 base.cell_vertices, base.cell_edges, base.basis_vectors, base_radius=2.)


class OutlineDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.set_scene(outline_scene())

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)

    def test_ring_preserves_solid_and_dashed_segments(self):
        mesh = self.viewer._atom_outline_mesh()
        self.assertEqual(mesh.shape, (128, 4))
        self.assertEqual(mesh.nbytes, 2048)
        for i in range(64):
            for endpoint in (0, 1):
                angle = 2 * np.pi * (i + endpoint) / 64
                dashed = 2 * np.pi * (i + (0 if i % 2 else .58 * endpoint)) / 64
                np.testing.assert_allclose(mesh[2*i+endpoint],
                    [np.cos(angle), np.sin(angle), np.cos(dashed), np.sin(dashed)], atol=3e-8)
        np.testing.assert_array_equal(mesh[2::4, 2:], mesh[3::4, 2:])

    def test_records_preserve_visibility_radius_colour_and_partial_occupancy(self):
        data = self.viewer._build_atom_outline_instances()
        self.assertEqual(data.shape, (3, 9))
        np.testing.assert_allclose(data[:, :3], self.viewer.scene.atom_centers)
        np.testing.assert_allclose(data[:, 3], [.3, .4, .3])
        np.testing.assert_array_equal(data[:, 8], [0, 0, 0])
        self.viewer.set_display_options(engraving=True, atom_scale=2, show_external_atoms=False)
        data = self.viewer._build_atom_outline_instances()
        np.testing.assert_allclose(data[:, 3], [.6, .8])
        np.testing.assert_allclose(data[:, 4:8], [[30/255]*3+[1]]*2)
        np.testing.assert_array_equal(data[:, 8], [0, 1])
        self.viewer.set_style_overrides(StyleOverrides(atom_component_visibility={'partial': False}))
        self.assertEqual(len(self.viewer._build_atom_outline_instances()), 1)
        self.viewer.set_display_options(show_atoms=False)
        self.assertEqual(self.viewer._build_atom_outline_instances().shape, (0, 9))
        self.viewer.set_scene(None)
        self.assertEqual(self.viewer._build_atom_outline_instances().shape, (0, 9))

    def test_camera_changes_and_mouse_rotation_do_not_invalidate_records(self):
        self.viewer._atom_outline_dirty = False
        before = self.viewer._build_atom_outline_instances()
        self.viewer.set_orientation(rotation(1, 43))
        self.viewer.zoom_by(2)
        with patch.object(renderer, 'glViewport'):
            self.viewer.resizeGL(800, 600)
        self.viewer._last_mouse = QPointF(50, 50)
        self.viewer._drag_button = Qt.MouseButton.LeftButton
        event = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(70, 60), QPointF(70, 60),
                            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.viewer.mouseMoveEvent(event)
        self.assertFalse(self.viewer._atom_outline_dirty)
        np.testing.assert_array_equal(before, self.viewer._build_atom_outline_instances())
        self.viewer.set_display_options(atom_scale=1.5)
        self.assertTrue(self.viewer._atom_outline_dirty)


class OpenGLOutlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen', 'minimal'}:
            raise unittest.SkipTest('A window system with OpenGL is required')

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.resize(640, 480)
        self.viewer.set_scene(outline_scene())
        self.viewer.set_display_options(engraving=True)
        self.viewer.show()
        self.app.processEvents()
        self.assertTrue(self.viewer.isValid())
        self.assertIsNone(self.viewer.opengl_error)

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)
        self.app.processEvents()

    def pixels(self):
        image = self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.viewer.opengl_error)
        self.assertFalse(image.isNull())
        return np.frombuffer(image.bits(), np.uint8, count=image.sizeInBytes()).copy()

    def test_camera_updates_without_outline_upload_and_style_restores_frame(self):
        before = self.pixels()
        with patch.object(self.viewer, '_rebuild_atom_outline_cache', wraps=self.viewer._rebuild_atom_outline_cache) as build, \
             patch.object(renderer, 'glDrawArraysInstanced', wraps=renderer.glDrawArraysInstanced) as draw:
            self.viewer.set_orientation(rotation(1, 33))
            self.viewer.zoom_by(2)
            self.viewer.resize(800, 600)
            self.app.processEvents()
            self.pixels()
            build.assert_not_called()
            self.assertTrue(draw.called)
            self.assertEqual(draw.call_args.args[1:], (0, 128, 3))
        self.viewer.resize(640, 480)
        self.viewer.set_orientation(np.eye(3))
        self.viewer.zoom_by(-2)
        self.app.processEvents()
        np.testing.assert_array_equal(self.pixels(), before)
        self.viewer.set_display_options(engraving=False, atom_scale=1.5)
        self.assertFalse(np.array_equal(self.pixels(), before))
        self.viewer.set_display_options(engraving=True, atom_scale=1)
        np.testing.assert_array_equal(self.pixels(), before)

    def test_buffer_layout_cleanup_and_context_reinitialization(self):
        from OpenGL.GL import (GL_ARRAY_BUFFER, GL_BUFFER_SIZE, GL_VERTEX_ATTRIB_ARRAY_DIVISOR,
                               glBindBuffer, glBindVertexArray, glGetBufferParameteriv,
                               glGetVertexAttribiv, glIsBuffer, glIsProgram)
        before = self.pixels()
        buffers = [self.viewer._atom_outline_vbo, self.viewer._atom_outline_mesh_vbo]
        program = self.viewer._atom_outline_program
        self.viewer.makeCurrent()
        try:
            glBindVertexArray(self.viewer._atom_outline_vao)
            for location in range(4):
                self.assertEqual(np.asarray(glGetVertexAttribiv(location, GL_VERTEX_ATTRIB_ARRAY_DIVISOR)).ravel()[0],
                                 0 if location == 0 else 1)
            for buffer, size in zip(buffers, (108, 2048)):
                glBindBuffer(GL_ARRAY_BUFFER, buffer)
                self.assertEqual(int(glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE)), size)
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        finally:
            self.viewer.doneCurrent()
        self.viewer.cleanup()
        self.viewer.makeCurrent()
        try:
            self.assertTrue(all(not glIsBuffer(buffer) for buffer in buffers))
            self.assertFalse(glIsProgram(program))
            self.viewer.initializeGL()
        finally:
            self.viewer.doneCurrent()
        np.testing.assert_array_equal(self.pixels(), before)
        self.viewer.set_display_options(show_atoms=False)
        with patch.object(renderer, 'glDrawArraysInstanced', wraps=renderer.glDrawArraysInstanced) as draw:
            self.pixels()
            draw.assert_not_called()
        self.viewer.set_display_options(show_atoms=True)
        np.testing.assert_array_equal(self.pixels(), before)
