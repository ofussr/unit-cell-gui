"""Procedural hatching must actually execute, preserve the ink, and release GL resources."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
from test_unit_cell_gui import rotation, sample_scene
from test_visibility_cache import repeated_scene

class GPUHatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}: raise unittest.SkipTest('OpenGL display required')
    def make_viewer(self, backend, scene):
        with patch.dict(os.environ, {'UNIT_CELL_GUI_HATCH_BACKEND':backend}):
            v=UnitCellViewer();v.resize(640,480);v.set_scene(scene)
            v.set_display_options(engraving=True,show_polyhedra=True,show_atoms=False,show_cell=False,show_basis=False)
            v.show();self.app.processEvents()
        self.addCleanup(self.close_viewer,v)
        return v
    def close_viewer(self,v):
        v.close();shiboken6.delete(v);self.app.processEvents()
    def pixels(self,v):
        image=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(v.opengl_error)
        self.assertFalse(image.isNull())
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)
    def test_shader_no_cpu_strokes_and_texture_cleanup(self):
        v=self.make_viewer('glsl',repeated_scene())
        with patch.object(v,'_build_polyhedron_hatch_geometry',side_effect=AssertionError('unexpected CPU stroke construction')):
            v.set_orientation(rotation(0,23)@rotation(1,37));self.pixels(v)
        self.assertTrue(v._hatch_gpu_program)
        self.assertIsNone(v.hatch_gpu_error)
        textures=[b.texture for b in v._hatch_buffers.values()];self.assertTrue(textures and all(textures))
        # Check deletion immediately, before Qt can recycle the texture names.
        original=v._delete_hatch_buffers
        def checked_delete():
            original()
            self.assertTrue(all(not GL.glIsTexture(t) for t in textures))
        with patch.object(v,'_delete_hatch_buffers',side_effect=checked_delete):
            v.set_scene(None);self.pixels(v)
        self.assertFalse(v._hatch_buffers)
    def test_reference_octants_and_blank_face(self):
        scene=sample_scene();cpu=self.make_viewer('python',scene);gpu=self.make_viewer('glsl',scene)
        z=np.array([1.,1.,1.])/np.sqrt(3);x=np.array([1.,-1.,0.])/np.sqrt(2);y=np.cross(z,x)
        poses=[np.eye(3),np.array([x,y,z]),rotation(0,17)@rotation(1,33),rotation(0,67)@rotation(1,21),rotation(1,90),np.diag([-1.,1.,1.]),np.array([[1.,.13,0],[.1,1.,.2],[0.,0.,1.]])]
        for pose in poses:
            for v in (cpu,gpu): v.set_orientation(pose)
            a=self.pixels(cpu);b=self.pixels(gpu)
            self.assertIsNone(gpu.hatch_gpu_error)
            d=np.abs(a.astype(np.int16)-b.astype(np.int16))[:,:,:3]
            # Rasterization at subpixel stroke ends can differ; reject misplaced or missing strokes.
            self.assertLess(float(d.mean()),.1)
            self.assertLess(float(np.any(d>10,axis=2).mean()),.005)
    def test_upload_failure_falls_back_to_cpu(self):
        v=self.make_viewer('glsl',sample_scene())
        with patch.object(v,'_upload_gpu_hatch_plan',side_effect=ValueError('forced limit')):
            v.set_orientation(rotation(0,37));image=self.pixels(v)
        self.assertEqual(v.hatch_gpu_error,'forced limit');self.assertTrue(all(not b.texture for b in v._hatch_buffers.values()))
        cpu=self.make_viewer('python',sample_scene());cpu.set_orientation(rotation(0,37))
        np.testing.assert_array_equal(image,self.pixels(cpu))
