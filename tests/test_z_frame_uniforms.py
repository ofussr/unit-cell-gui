"""UBO and native state must be verified independently of draw submission."""
import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL as g
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer, opengl_viewer as renderer
from unit_cell_gui._frame_uniforms import SLOT
from unit_cell_gui._weighted_oit import preserved_state
from test_unit_cell_gui import sample_scene, rotation

class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('OpenGL display required')
    def viewer(self,backend='rust',fail=False):
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,{'UNIT_CELL_GUI_FRAME_BACKEND':backend}))
            if fail:stack.enter_context(patch.object(renderer,'FrameParameters',side_effect=RuntimeError('test UBO failure')))
            v=UnitCellViewer();v.resize(480,360);v.set_scene(sample_scene());v.set_display_options(show_polyhedra=True);v.show();self.app.processEvents()
        self.addCleanup(self.close_viewer,v)
        self.assertIsNone(v.opengl_error)
        return v
    def close_viewer(self,v):
        v.close();shiboken6.delete(v);self.app.processEvents()
    def pixels(self,v):
        im=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertFalse(im.isNull());self.assertIsNone(v.opengl_error);self.assertIsNone(v.weighted_oit_error)
        return np.frombuffer(im.bits(),np.uint8,count=im.sizeInBytes()).copy().reshape(im.height(),im.width(),4)
    def test_exact_weighted_engraving_reference(self):
        a,b=self.viewer(),self.viewer('python')
        self.assertIsNotNone(a._frame_parameters);self.assertIsNone(a.frame_uniform_error);self.assertIsNone(b._frame_parameters)
        for engraving,weighted in ((False,False),(False,True),(True,False)):
            for v in (a,b):
                v.set_display_options(engraving=engraving,weighted_blended_oit=weighted)
                v.set_orientation(rotation(0,23)@rotation(1,37));v.set_polyhedron_opacities({0:.4})
            np.testing.assert_array_equal(self.pixels(a),self.pixels(b))
            self.assertEqual(a._last_frame_wboit,weighted)
    def test_one_upload_on_camera_change_and_no_legacy_matrix_uniforms(self):
        v=self.viewer();f=v._frame_parameters;self.assertIsNotNone(f)
        self.pixels(v);count=f.uploads
        with patch.object(renderer,'glUniformMatrix3fv',side_effect=AssertionError('legacy matrix')),patch.object(renderer,'glUniformMatrix4fv',side_effect=AssertionError('legacy matrix')):
            self.pixels(v);self.assertEqual(f.uploads,count)
            v.set_orientation(rotation(1,19));self.pixels(v);self.assertEqual(f.uploads,count+1)
            v.set_display_options(weighted_blended_oit=True);self.pixels(v);self.assertTrue(v._last_frame_wboit)
            self.assertEqual(f.uploads,count+1)
            v.pan_x+=7;self.pixels(v);self.assertEqual(f.uploads,count+2)
        v.makeCurrent()
        try:
            self.assertEqual(g.glGetUniformLocation(v._program,'u_projection'),-1)
            self.assertNotEqual(int(g.glGetUniformBlockIndex(v._program,'UcgFrame')),0xffffffff)
        finally:v.doneCurrent()
    def test_ubo_restores_foreign_range_even_after_exception(self):
        v=self.viewer();f=v._frame_parameters;v.makeCurrent()
        buffers=list(map(int,g.glGenBuffers(2)));align=int(g.glGetIntegerv(g.GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT))
        try:
            for b in buffers:g.glBindBuffer(g.GL_UNIFORM_BUFFER,b);g.glBufferData(g.GL_UNIFORM_BUFFER,align*4,None,g.GL_DYNAMIC_DRAW)
            g.glBindBufferRange(g.GL_UNIFORM_BUFFER,SLOT,buffers[0],align,align*2)
            g.glBindBuffer(g.GL_UNIFORM_BUFFER,buffers[1])
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with f.bound(v):
                    self.assertEqual(int(g.glGetIntegeri_v(g.GL_UNIFORM_BUFFER_BINDING,SLOT)),f.buffer)
                    raise RuntimeError('injected')
            self.assertEqual(int(g.glGetIntegerv(g.GL_UNIFORM_BUFFER_BINDING)),buffers[1])
            self.assertEqual(int(g.glGetIntegeri_v(g.GL_UNIFORM_BUFFER_BINDING,SLOT)),buffers[0])
            # Native capture exposes offset and size without PyOpenGL's int64 conversion.
            state=f.native.bind_frame(f.buffer,None);f.native.restore_frame(state)
            self.assertEqual(state,(buffers[1],buffers[0],align,align*2))
            self.assertEqual(g.glGetError(),g.GL_NO_ERROR)
        finally:
            g.glBindBufferBase(g.GL_UNIFORM_BUFFER,SLOT,0);g.glBindBuffer(g.GL_UNIFORM_BUFFER,0)
            g.glDeleteBuffers(len(buffers),buffers);v.doneCurrent()
    def test_native_state_matches_legacy_and_restores_on_exception(self):
        v=self.viewer();v.makeCurrent();native=v._frame_parameters.native
        try:
            with preserved_state():
                g.glEnable(g.GL_BLEND);g.glDepthMask(False);g.glColorMask(True,False,True,False)
                g.glActiveTexture(g.GL_TEXTURE2);g.glViewport(1,2,123,97)
                before=native.save_state()
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    with preserved_state(native):
                        g.glDisable(g.GL_BLEND);g.glDepthMask(True);g.glColorMask(False,True,False,True)
                        g.glActiveTexture(g.GL_TEXTURE0);g.glViewport(0,0,9,10)
                        raise RuntimeError('injected')
                self.assertEqual(native.save_state(),before)
                self.assertEqual(g.glGetError(),g.GL_NO_ERROR)
        finally:v.doneCurrent()
    def test_straight_alpha_blend_preserves_opaque_destination_alpha(self):
        for backend in ('rust','python'):
            v=self.viewer(backend);v.makeCurrent()
            try:
                with preserved_state():
                    v._blend_pass(True)
                    self.assertEqual(int(g.glGetIntegerv(g.GL_BLEND_SRC_RGB)),g.GL_SRC_ALPHA)
                    self.assertEqual(int(g.glGetIntegerv(g.GL_BLEND_DST_RGB)),g.GL_ONE_MINUS_SRC_ALPHA)
                    self.assertEqual(int(g.glGetIntegerv(g.GL_BLEND_SRC_ALPHA)),g.GL_ONE)
                    self.assertEqual(int(g.glGetIntegerv(g.GL_BLEND_DST_ALPHA)),g.GL_ONE_MINUS_SRC_ALPHA)
                    v._blend_pass(False)
            finally:v.doneCurrent()

    def test_allocation_failure_fallback_and_context_recreation(self):
        fallback=self.viewer(fail=True);self.assertIsNone(fallback._frame_parameters)
        self.assertEqual(fallback.frame_uniform_error,'test UBO failure');self.pixels(fallback)
        v=self.viewer();initial=self.pixels(v);f=v._frame_parameters
        v.close();self.app.processEvents();self.assertEqual(f.buffer,0);self.assertIsNone(v._frame_parameters)
        with patch.dict(os.environ,{'UNIT_CELL_GUI_FRAME_BACKEND':'rust'}):v.show();self.app.processEvents()
        self.assertIsNotNone(v._frame_parameters);self.assertIsNot(v._frame_parameters,f)
        np.testing.assert_array_equal(initial,self.pixels(v))
