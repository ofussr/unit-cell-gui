"""Beta review regressions: context ownership, GL state and fallback paths."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from unit_cell_gui import UnitCellViewer, _gl
from unit_cell_gui._weighted_oit import WeightedOIT, preserved_state
from test_unit_cell_gui import sample_scene, rotation

class BindingTests(unittest.TestCase):
    def test_invalid_payloads_and_counts_rejected_before_native_calls(self):
        with patch.object(_gl, '_call') as call:
            for args in ((0,-1,None,0),(0,3,b'1234',0)):
                with self.assertRaises(ValueError): _gl.glBufferData(*args)
            for args in ((0,-1,4,b'1234'),(0,0,4,None),(0,0,2,b'1234')):
                with self.assertRaises(ValueError): _gl.glBufferSubData(*args)
            with self.assertRaises(ValueError): _gl.glDeleteBuffers(2,[1])
            call.assert_not_called()

class BetaGLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('GL display required')
    def viewer(self, settings=None):
        with patch.dict(os.environ, settings or {}):
            v=UnitCellViewer();self.addCleanup(self.close_viewer,v)
            v.resize(321,241);v.set_scene(sample_scene());v.set_display_options(show_polyhedra=True)
            v.show();self.app.processEvents();self.pixels(v)
        return v
    def close_viewer(self,v):
        v.close();shiboken6.delete(v);self.app.processEvents()
    def pixels(self,v):
        image=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertFalse(image.isNull());self.assertIsNone(v.opengl_error);self.assertIsNone(v.weighted_oit_error)
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy()
    def test_cleanup_without_current_context_does_not_discard_handles(self):
        v=self.viewer();program=v._program;resources=v._resource_gl;v.doneCurrent()
        with patch.object(v,'makeCurrent',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'context current'):v.cleanup()
        self.assertEqual(v._program,program);self.assertIs(v._resource_gl,resources)
        self.assertFalse(v._cleaning_up)
        v.makeCurrent()
        try:self.assertTrue(GL.glIsProgram(program))
        finally:v.doneCurrent()
        v.cleanup();self.assertEqual(v._program,0)
    def test_allocation_preserves_host_texture_unit_in_both_resource_backends(self):
        for backend in ('rust','python'):
            v=self.viewer({'UNIT_CELL_GUI_RESOURCE_BACKEND':backend});v.makeCurrent()
            unit=int(GL.GL_TEXTURE0)+5;previous=int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE))
            GL.glActiveTexture(unit);old=int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D));texture=int(GL.glGenTextures(1));GL.glBindTexture(GL.GL_TEXTURE_2D,texture)
            try:
                oit=WeightedOIT(v._resource_gl)
                with preserved_state(v._frame_parameters.native if v._frame_parameters is not None else None):
                    oit.ensure_size(48,48,1)
                self.assertEqual(int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)),unit)
                self.assertEqual(int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)),texture)
                with preserved_state():oit.cleanup()
            finally:
                GL.glActiveTexture(unit);GL.glBindTexture(GL.GL_TEXTURE_2D,old);GL.glDeleteTextures(1,[texture]);GL.glActiveTexture(previous);v.doneCurrent()
    def test_legacy_byte_upload_exact_and_empty_allocation(self):
        v=self.viewer();v.makeCurrent();b=int(GL.glGenBuffers(1))
        try:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER,b)
            _gl.glBufferData(GL.GL_ARRAY_BUFFER,4,b'\x00\xff\x00\x00',GL.GL_DYNAMIC_DRAW)
            _gl.glBufferSubData(GL.GL_ARRAY_BUFFER,1,2,memoryview(b'\x11\x22'))
            self.assertEqual(bytes(GL.glGetBufferSubData(GL.GL_ARRAY_BUFFER,0,4)),b'\x00\x11\x22\x00')
            _gl.glBufferData(GL.GL_ARRAY_BUFFER,0,None,GL.GL_STATIC_DRAW)
            self.assertEqual(int(GL.glGetBufferParameteriv(GL.GL_ARRAY_BUFFER,GL.GL_BUFFER_SIZE)),0)
        finally:GL.glBindBuffer(GL.GL_ARRAY_BUFFER,0);GL.glDeleteBuffers(1,[b]);v.doneCurrent()
    def test_all_eight_reference_backends_can_work_together(self):
        settings={f'UNIT_CELL_GUI_{key}_BACKEND':'python' for key in ('SORT','HATCH','CONTACT','RENDER','FRAME','RESOURCE')}
        settings.update(UNIT_CELL_GUI_ATOM_BACKEND='mesh',UNIT_CELL_GUI_WBOIT_BACKEND='dual')
        with patch.dict(os.environ,settings):
            v=self.viewer();self.assertIsNone(v._frame_parameters);self.assertIsNone(v._resource_gl.native);self.assertIsNone(v._native_gl)
            for engraving,weighted in ((False,False),(True,False),(False,True)):
                v.set_display_options(engraving=engraving,weighted_blended_oit=weighted)
                v.set_orientation(rotation(0,23)@rotation(1,37));v.set_atom_positions({0:[.05,.03,.01]});v.set_polyhedron_opacities({0:.4})
                image=self.pixels(v);self.assertGreater(len(np.unique(image)),8);self.assertEqual(v._last_frame_wboit,weighted)
            self.assertFalse(v._weighted_oit.single_pass)
