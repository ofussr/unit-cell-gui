"""Independent blend path, fallback and no-PyOpenGL runtime regressions."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
from unit_cell_gui import _gl as binding
from unit_cell_gui._weighted_oit import WeightedOIT
from test_unit_cell_gui import sample_scene, rotation

class SingleWBOITTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('OpenGL display required')
    def viewer(self,backend='auto',samples=4):
        with patch.dict(os.environ,{'UNIT_CELL_GUI_WBOIT_BACKEND':backend}):
            v=UnitCellViewer();fmt=v.format();fmt.setSamples(samples);v.setFormat(fmt)
            v.resize(321,241);v.set_scene(sample_scene());v.set_display_options(show_polyhedra=True,weighted_blended_oit=True)
            v.set_orientation(rotation(0,23)@rotation(1,37));v.set_polyhedron_opacities({0:.4})
            v.show();self.app.processEvents();self.pixels(v)
        self.addCleanup(self.close_viewer,v)
        return v
    def close_viewer(self,v):
        v.close();shiboken6.delete(v);self.app.processEvents()
    def pixels(self,v):
        im=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertFalse(im.isNull());self.assertIsNone(v.opengl_error);self.assertIsNone(v.weighted_oit_error)
        self.assertTrue(v._last_frame_wboit)
        return np.frombuffer(im.bits(),np.uint8,count=im.sizeInBytes()).copy().reshape(im.height(),im.width(),4)
    def require_single(self,v):
        if v._weighted_oit.indexed is None:
            self.skipTest('Independent blend not supported: '+str(v._weighted_oit.single_pass_error))
        self.assertTrue(v._weighted_oit.single_pass,v._weighted_oit.single_pass_error)
    def test_single_matches_dual_and_submits_once_with_and_without_msaa(self):
        for samples in (0,4):
            a=self.viewer('auto',samples);b=self.viewer('dual',samples)
            self.require_single(a);self.assertFalse(b._weighted_oit.single_pass)
            self.assertEqual(a._weighted_oit.size,b._weighted_oit.size)
            with patch.object(a,'_draw_weighted_ranges',wraps=a._draw_weighted_ranges) as draw:
                actual=self.pixels(a)
                nonempty=[c for c in draw.call_args_list if c.args[0] and c.args[2] in (a._weighted_oit.triangle_program,a._weighted_oit.segment_program)]
            with patch.object(b,'_draw_weighted_ranges',wraps=b._draw_weighted_ranges) as draw:
                expected=self.pixels(b)
                old=[c for c in draw.call_args_list if c.args[0] and c.args[2] in (b._weighted_oit.triangle_program,b._weighted_oit.segment_program)]
            self.assertGreater(len(nonempty),0);self.assertEqual(len(old),2*len(nonempty))
            np.testing.assert_allclose(actual,expected,atol=1)
    def test_independent_state_restored_after_exception(self):
        v=self.viewer();self.require_single(v);v.makeCurrent()
        try:
            oit=v._weighted_oit;native=oit.indexed;original=native.save()
            try:
                custom=[list(row) for row in original]
                custom[1]=[int(GL.GL_SRC_ALPHA),int(GL.GL_ONE),int(GL.GL_ONE),int(GL.GL_ZERO),int(GL.GL_FUNC_REVERSE_SUBTRACT),int(GL.GL_FUNC_ADD),0,1,0,1,0]
                native.restore(custom)
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    with oit.preserve_indexed():
                        native.configure();raise RuntimeError('injected')
                self.assertEqual(native.save(),custom)
            finally:native.restore(original)
            self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:v.doneCurrent()
    def test_combined_framebuffer_failure_retries_dual(self):
        with patch.object(binding,'glDrawBuffers',side_effect=RuntimeError('injected MRT failure')):
            a=self.viewer()
        self.assertFalse(a._weighted_oit.single_pass)
        if a._weighted_oit.indexed is None:self.skipTest('Independent blending not supported')
        self.assertIn('injected MRT failure',a._weighted_oit.single_pass_error)
        b=self.viewer('dual')
        self.assertFalse(b._weighted_oit.single_pass)
        self.assertEqual(a._weighted_oit.size,b._weighted_oit.size)
        # Inspect the actual draw targets, rather than relying on the flag alone.
        for v in (a,b):
            v.makeCurrent()
            try:
                oit=v._weighted_oit
                previous=int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
                try:
                    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER,oit.framebuffers[1])
                    self.assertEqual(GL.glCheckFramebufferStatus(GL.GL_DRAW_FRAMEBUFFER),GL.GL_FRAMEBUFFER_COMPLETE)
                    self.assertEqual(int(GL.glGetIntegerv(GL.GL_DRAW_BUFFER0)),GL.GL_COLOR_ATTACHMENT0)
                    self.assertEqual(int(GL.glGetIntegerv(GL.GL_DRAW_BUFFER1)),GL.GL_NONE)
                    self.assertEqual(int(GL.glGetFramebufferAttachmentParameteriv(
                        GL.GL_DRAW_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT1,
                        GL.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE)),GL.GL_NONE)
                    self.assertEqual(int(GL.glGetFramebufferAttachmentParameteriv(
                        GL.GL_DRAW_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,
                        GL.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME)),oit.textures[1])
                    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER,oit.framebuffers[2])
                    self.assertEqual(GL.glCheckFramebufferStatus(GL.GL_DRAW_FRAMEBUFFER),GL.GL_FRAMEBUFFER_COMPLETE)
                    self.assertEqual(int(GL.glGetFramebufferAttachmentParameteriv(
                        GL.GL_DRAW_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,
                        GL.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME)),oit.textures[2])
                finally:
                    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER,previous)
                self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
            finally:
                v.doneCurrent()
        actual,expected=self.pixels(a),self.pixels(b)
        # Independent Qt contexts and multisample resolves can differ by one
        # 8-bit colour level on some drivers (observed on Windows).
        # Keep exact alpha and a strict absolute per-channel RGB bound.
        self.assertEqual(actual.shape,expected.shape)
        np.testing.assert_array_equal(actual[:,:,3],expected[:,:,3])
        np.testing.assert_allclose(actual[:,:,:3],expected[:,:,:3],rtol=0,atol=1)
    def test_runtime_without_pyopengl(self):
        # Independent GL probes in the suite use PyOpenGL; the library must not.
        script='''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'OpenGL' or fullname.startswith('OpenGL.'):
            raise ImportError('PyOpenGL deliberately unavailable')
sys.meta_path.insert(0, Block())
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
from test_unit_cell_gui import sample_scene, rotation
app=QApplication([])
v=UnitCellViewer();v.resize(320,240);v.set_scene(sample_scene());v.show();app.processEvents()
for engraving,weighted in ((False,False),(False,True),(True,False)):
    v.set_display_options(show_polyhedra=True,engraving=engraving,weighted_blended_oit=weighted)
    v.set_orientation(rotation(0,23)@rotation(1,37));image=v.grabFramebuffer()
    assert not image.isNull() and v.opengl_error is None and v.weighted_oit_error is None, (v.opengl_error,v.weighted_oit_error)
    assert v._last_frame_wboit == weighted
assert not any(n=='OpenGL' or n.startswith('OpenGL.') for n in sys.modules)
v.close()
'''
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([os.path.dirname(__file__),env.get('PYTHONPATH','')])
        result=subprocess.run([sys.executable,'-c',script],env=env,capture_output=True,text=True,timeout=45)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
