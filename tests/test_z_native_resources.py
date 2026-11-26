"""Verify native resource paths, byte-exact updates and ownership."""
import os
from unit_cell_gui import _gl as gl_binding
import unittest
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL as g
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer, opengl_viewer as renderer
from unit_cell_gui._native_resources import ResourceGL
from unit_cell_gui._weighted_oit import WeightedOIT, preserved_state
from test_unit_cell_gui import sample_scene, rotation

class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('OpenGL display required')
    def setUp(self):self.viewers=[]
    def tearDown(self):
        for v in self.viewers:v.close();shiboken6.delete(v);self.app.processEvents()
    def viewer(self,backend='rust'):
        with patch.dict(os.environ,{'UNIT_CELL_GUI_RESOURCE_BACKEND':backend}):
            v=UnitCellViewer();self.viewers.append(v);v.resize(480,360);v.set_scene(sample_scene());v.set_display_options(show_polyhedra=True);v.show();self.app.processEvents();self.pixels(v)
        self.assertEqual(v._resource_gl.native is not None,backend=='rust',v.resource_error)
        return v
    def pixels(self,v):
        im=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(v.opengl_error);self.assertIsNone(v.weighted_oit_error);self.assertFalse(im.isNull())
        return np.frombuffer(im.bits(),np.uint8,count=im.sizeInBytes()).copy()
    def test_native_resource_path_does_not_call_legacy_resource_operations(self):
        with ExitStack() as stack:
            for name in ('glGenBuffers','glGenVertexArrays','glGenTextures','glBufferData','glBufferSubData','glTexBuffer','glDeleteBuffers','glDeleteTextures','glDeleteVertexArrays','glVertexAttribPointer','glVertexAttribIPointer'):
                stack.enter_context(patch.object(renderer,name,side_effect=AssertionError(name)))
            for name in ('glGenFramebuffers','glGenRenderbuffers','glTexImage2D','glTexImage2DMultisample','glRenderbufferStorage','glRenderbufferStorageMultisample','glFramebufferTexture2D','glFramebufferRenderbuffer','glDeleteFramebuffers','glDeleteRenderbuffers'):
                stack.enter_context(patch.object(gl_binding,name,side_effect=AssertionError(name)))
            v=self.viewer();r=v._resource_gl
            v.set_atom_positions({0:v.get_atom_positions()[0]+[.05,.03,0]});self.pixels(v)
            self.assertGreater(r.patches,0)
            v.set_display_options(engraving=True);self.pixels(v)
            v.set_display_options(engraving=False,weighted_blended_oit=True);self.pixels(v);self.assertTrue(v._last_frame_wboit)
            v.resize(510,400);self.pixels(v)
            v.close();self.app.processEvents()
    def test_disjoint_patches_are_byte_exact_and_invalid_requests_do_not_write(self):
        v=self.viewer();r=v._resource_gl;v.makeCurrent();b=r.glGenBuffers(1);vao=r.glGenVertexArrays(1)
        try:
            a=np.arange(80,dtype=np.float32).reshape(20,4);r.upload_static(vao,b,a)
            changes=[(16,a[1:2]+100),(48,a[3:5]+200),(304,a[19:20]+300)]
            expected=a.copy();expected[1:2]+=100;expected[3:5]+=200;expected[19:20]+=300
            p,size=r.patches,r.patch_bytes;r.patch_rows(b,changes)
            self.assertEqual((r.patches-p,r.patch_bytes-size),(3,64))
            def read():
                g.glBindBuffer(g.GL_ARRAY_BUFFER,b);data=g.glGetBufferSubData(g.GL_ARRAY_BUFFER,0,a.nbytes);g.glBindBuffer(g.GL_ARRAY_BUFFER,0);return np.frombuffer(data,dtype=np.float32).reshape(a.shape)
            np.testing.assert_array_equal(read(),expected)
            with self.assertRaises(ValueError):r.native.patch_rows(b,[(0,b'1234'),(-1,b'1234')])
            with self.assertRaises(ValueError):r.native.buffer_data(int(g.GL_ARRAY_BUFFER),3,int(g.GL_STATIC_DRAW),b'1234')
            np.testing.assert_array_equal(read(),expected)
            self.assertEqual(int(g.glGetIntegerv(g.GL_ARRAY_BUFFER_BINDING)),0)
        finally:r.glDeleteBuffers(1,[b]);r.glDeleteVertexArrays(1,[vao]);v.doneCurrent()
    def test_context_guard_and_resource_recreation(self):
        v=self.viewer();r=v._resource_gl;old=self.pixels(v)
        v.doneCurrent()
        with self.assertRaisesRegex(RuntimeError,'not current'):r.glGenBuffers(1)
        v.close();self.app.processEvents()
        self.assertIsNone(v._resource_gl.native)
        with patch.dict(os.environ,{'UNIT_CELL_GUI_RESOURCE_BACKEND':'rust'}):v.show();self.app.processEvents()
        self.assertIsNot(v._resource_gl,r);self.assertIsNotNone(v._resource_gl.native)
        np.testing.assert_array_equal(old,self.pixels(v))
    def test_wboit_attachment_allocation_and_release_native(self):
        v=self.viewer();r=v._resource_gl;v.makeCurrent()
        try:
            with preserved_state():
                oit=WeightedOIT(r)
                for samples in (1,4):
                    oit.ensure_size(48,48,samples);handles=(list(oit.textures),list(oit.framebuffers),oit.depth)
                    self.assertTrue(all(g.glIsTexture(t) for t in handles[0]));self.assertTrue(all(g.glIsFramebuffer(t) for t in handles[1]));self.assertTrue(g.glIsRenderbuffer(handles[2]))
                    oit.release_attachments();self.assertTrue(all(not g.glIsTexture(t) for t in handles[0]));self.assertTrue(all(not g.glIsFramebuffer(t) for t in handles[1]));self.assertFalse(g.glIsRenderbuffer(handles[2]))
        finally:v.doneCurrent()
    def test_backend_identical_with_local_updates_and_modes(self):
        a,b=self.viewer(),self.viewer('python')
        for engraving,weighted in ((False,False),(True,False),(False,True)):
            for v in (a,b):
                v.set_display_options(engraving=engraving,weighted_blended_oit=weighted)
                v.set_orientation(rotation(0,23)@rotation(1,37));v.set_atom_positions({0:[.05,.03,.01]});v.set_polyhedron_opacities({0:.4})
            np.testing.assert_array_equal(self.pixels(a),self.pixels(b))
    def test_missing_entry_point_falls_back_before_any_allocation(self):
        v=self.viewer();v.makeCurrent()
        try:
            from unit_cell_gui import _native_resources
            with patch.object(_native_resources,'_SYMBOLS',('not_a_gl_function_ucg',)):
                r=ResourceGL(v.context(),renderer)
            self.assertIsNone(r.native);self.assertTrue(r.error)
            b=r.glGenBuffers(1);r.glDeleteBuffers(1,[b])
        finally:v.doneCurrent()
    def test_copy_fallback_and_noncontiguous_inputs(self):
        v=self.viewer();r=v._resource_gl;v.makeCurrent();b=r.glGenBuffers(1);vao=r.glGenVertexArrays(1)
        try:
            a=np.arange(128,dtype=np.float32).reshape(16,8)[:,::2]
            for borrowed in (False,r.borrowed_buffers):
                with patch.object(r,'borrowed_buffers',borrowed):
                    r.upload_static(vao,b,a)
                    g.glBindBuffer(g.GL_ARRAY_BUFFER,b)
                    got=g.glGetBufferSubData(g.GL_ARRAY_BUFFER,0,a.nbytes)
                    g.glBindBuffer(g.GL_ARRAY_BUFFER,0)
                    np.testing.assert_array_equal(np.frombuffer(got,np.float32).reshape(a.shape),a)
        finally:r.glDeleteBuffers(1,[b]);r.glDeleteVertexArrays(1,[vao]);v.doneCurrent()
    def test_native_allocation_failure_releases_registered_handles(self):
        v=self.viewer();r=v._resource_gl;v.makeCurrent()
        try:
            with preserved_state():
                oit=WeightedOIT(r);textures=[];depth=[]
                t=r.glGenTextures;d=r.glGenRenderbuffers
                def texture(n):
                    handle=t(n);textures.append(handle);return handle
                def renderbuffer(n):
                    handle=d(n);depth.append(handle);return handle
                # Fail framebuffer generation after the first texture storage
                # is configured, then verify every registered name is released.
                with patch.object(r,'glGenTextures',side_effect=texture),patch.object(r,'glGenRenderbuffers',side_effect=renderbuffer),patch.object(r,'glGenFramebuffers',side_effect=RuntimeError('injected')):
                    with self.assertRaisesRegex(RuntimeError,'injected'):oit.ensure_size(48,48,1)
                self.assertEqual(len(textures),1);self.assertEqual(len(depth),1)
                self.assertTrue(all(not g.glIsTexture(t) for t in textures));self.assertTrue(all(not g.glIsRenderbuffer(t) for t in depth))
                self.assertEqual((oit.textures,oit.framebuffers,oit.depth),([],[],0))
        finally:v.doneCurrent()
