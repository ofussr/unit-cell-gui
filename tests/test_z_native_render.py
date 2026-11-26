"""Native submission must exercise the real Qt context and match the reference."""
import os
from unit_cell_gui import _gl as gl_binding
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer, StyleOverrides
from render_test_helpers import adjacent_cubes
from test_unit_cell_gui import sample_scene, rotation

class NativeRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}: raise unittest.SkipTest('OpenGL display required')
    def make_viewer(self, scene):
        with patch.dict(os.environ, {'UNIT_CELL_GUI_RENDER_BACKEND':'rust'}):
            v=UnitCellViewer();v.resize(480,360);v.set_scene(scene)
            v.set_display_options(show_polyhedra=True,show_atoms=True,show_cell=False,show_basis=False)
            v.show();self.app.processEvents()
        self.addCleanup(self.close_viewer,v)
        self.assertIsNone(v.opengl_error);self.assertIsNone(v.native_render_error)
        self.assertIsNotNone(v._native_gl)
        return v
    def close_viewer(self,v):
        v.close();shiboken6.delete(v);self.app.processEvents()
    def pixels(self,v):
        image=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(v.opengl_error);self.assertFalse(image.isNull())
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)
    def compare(self,v):
        native=self.pixels(v); self.assertGreater(v._native_draw_calls,0)
        with patch.object(v,'_native_gl',None):
            reference=self.pixels(v);self.assertEqual(v._native_draw_calls,0)
        np.testing.assert_array_equal(native,reference)
    def test_exact_and_weighted_draws_without_python_primitive_dispatch(self):
        v=self.make_viewer(adjacent_cubes(12));v.set_orientation(rotation(0,23)@rotation(1,37))
        for weighted in (False,True):
            v.set_display_options(weighted_blended_oit=weighted)
            self.compare(v)
            with patch.object(v,'_draw_primitive_instances',side_effect=AssertionError('unexpected Python submission')):
                self.pixels(v)
                self.assertGreater(v._native_draw_calls,0)
            self.assertEqual(v._last_frame_wboit,weighted)
            self.assertIsNone(v.weighted_oit_error)
    def test_chunked_and_partial_geometry(self):
        v=self.make_viewer(adjacent_cubes(6));v._primitive_texel_capacity=16
        v.set_scene(adjacent_cubes(6));v.set_orientation(rotation(0,31)@rotation(1,17))
        self.compare(v)
        self.assertTrue(any(len(c)>1 for c in v._primitive_chunks.values()))
        v.set_polyhedron_opacities({0:1.,1:.25,2:0.})
        self.compare(v)
    def test_contacts_and_context_recreation(self):
        v=self.make_viewer(sample_scene());v.set_orientation(rotation(0,23)@rotation(1,31))
        self.compare(v)
        v.set_polyhedron_opacities({0:.5});self.compare(v)
        v.close();self.app.processEvents();self.assertIsNone(v._native_gl)
        with patch.dict(os.environ, {'UNIT_CELL_GUI_RENDER_BACKEND':'rust'}):
            v.show();self.app.processEvents()
        self.assertIsNotNone(v._native_gl);self.compare(v)
    def test_validation_precedes_gl_dispatch(self):
        v=self.make_viewer(adjacent_cubes(2));v.makeCurrent()
        try:
            with self.assertRaises(ValueError):
                v._native_gl.draw_ranges([(1,1,1,4,[(1,2)])],[(0,0,3)],True)
            self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:v.doneCurrent()
    def test_initialization_failure_uses_python(self):
        with patch.dict(os.environ,{'UNIT_CELL_GUI_RENDER_BACKEND':'rust'}),patch('unit_cell_gui._native_render.create_submission',side_effect=RuntimeError('test resolver failure')):
            v=UnitCellViewer();v.resize(480,360);v.set_scene(adjacent_cubes(2));v.set_display_options(show_polyhedra=True)
            v.show();self.app.processEvents()
        self.addCleanup(self.close_viewer,v)
        self.assertIsNone(v._native_gl);self.assertEqual(v.native_render_error,'test resolver failure');self.pixels(v)
        self.assertEqual(v._native_draw_calls,0)
    def test_all_scene_modes_submit_draws_in_rust(self):
        from contextlib import ExitStack
        from unit_cell_gui import opengl_viewer as renderer
        v=self.make_viewer(sample_scene());v.set_orientation(rotation(0,23)@rotation(1,31))
        for engraving,weighted in ((False,False),(False,True),(True,False)):
            v.set_display_options(engraving=engraving,weighted_blended_oit=weighted)
            self.compare(v)
            with ExitStack() as stack:
                for name in ('glDrawArrays','glDrawArraysInstanced','glDrawElementsInstanced'):
                    stack.enter_context(patch.object(renderer,name,side_effect=AssertionError('unexpected Python '+name)))
                stack.enter_context(patch.object(gl_binding,'glDrawArrays',side_effect=AssertionError('unexpected Python composition')))
                self.pixels(v)
                self.assertGreater(v._native_draw_calls,0)
                self.assertEqual(v._last_frame_wboit,weighted)
                self.assertIsNone(v.weighted_oit_error)
    def test_foreign_context_is_never_used(self):
        a=self.make_viewer(adjacent_cubes(2));b=self.make_viewer(adjacent_cubes(2))
        b.makeCurrent()
        try:
            self.assertIsNone(a._native_submission())
            self.assertIsNotNone(b._native_submission())
        finally:b.doneCurrent()
