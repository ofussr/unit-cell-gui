"""Analytic sphere coverage, depth, materials, fallback and native resource tests."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import Atom, AtomComponent, Scene, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from test_unit_cell_gui import rotation, sample_scene
from render_test_helpers import face
from dataclasses import replace


def atoms_scene(mixed=False, positions=((0.,0.,0.),)):
    red=AtomComponent('r','r','C',.4 if mixed else 1.,'#c04020',.5)
    blue=AtomComponent('b','b','O',.35,'#3040d0',.5)
    atoms=tuple(Atom(str(i),str(i),(red,blue) if mixed else (red,)) for i in range(len(positions)))
    return Scene(atoms,positions,[],[],[],np.eye(3),base_radius=2.)


class SphereViewer(UnitCellViewer):
    # Isolate the actual sphere coverage/depth from intentional ink and labels.
    def _draw_overlay(self): pass
    def _draw_atom_outlines(self): pass


class AnalyticAtomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('OpenGL display required')

    def viewer(self,backend='glsl',samples=4,mixed=False,cls=SphereViewer,fail=False):
        v=cls();v.resize(512,512);fmt=v.format();fmt.setSamples(samples);v.setFormat(fmt)
        v.set_scene(atoms_scene(mixed));v.set_display_options(show_cell=False,show_basis=False,show_bonds=False,show_polyhedra=False)
        orig=renderer._create_program
        def compile_program(*args,**kwargs):
            if fail and args and 'sample out vec2 v_offset' in args[0]:raise RuntimeError('forced analytic compile failure')
            return orig(*args,**kwargs)
        with patch.dict(os.environ,{'UNIT_CELL_GUI_ATOM_BACKEND':backend}),patch.object(renderer,'_create_program',side_effect=compile_program):
            v.show();self.app.processEvents()
        self.addCleanup(self.close_viewer,v)
        self.assertIsNone(v.opengl_error)
        return v

    def close_viewer(self,v):
        v.close();shiboken6.delete(v);self.app.processEvents()

    def pixels(self,v):
        image=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertFalse(image.isNull());self.assertIsNone(v.opengl_error)
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)

    def require_analytic(self,v):
        fmt=v.context().format()
        if (fmt.majorVersion(),fmt.minorVersion())<(4,0) and not (v.context().hasExtension(b'GL_ARB_gpu_shader5') and v.context().hasExtension(b'GL_ARB_sample_shading')):
            self.skipTest('Sample-frequency shading unavailable; legacy fallback')
        self.assertIsNone(v.atom_analytic_error)
        self.assertGreater(v._atom_analytic_program,0)

    def test_exact_single_sample_depth_and_coverage(self):
        v=self.viewer(samples=0);self.require_analytic(v);img=self.pixels(v)
        self.assertEqual(v._last_atom_backend,'glsl')
        v.makeCurrent()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject())
            self.assertEqual(int(GL.glGetIntegerv(GL.GL_SAMPLES)),0)
            w,h=img.shape[1],img.shape[0];scale=v._screen_scale()*v.devicePixelRatioF();r=.5
            for dx,dy in ((0,0),(12,19),(-28,7),(0,40),(70,0)):
                x,y=w//2+dx,h//2+dy
                actual=float(np.asarray(GL.glReadPixels(x,y,1,1,GL.GL_DEPTH_COMPONENT,GL.GL_FLOAT)).ravel()[0])
                rho2=((x+.5-w/2)**2+(y+.5-h/2)**2)/(scale*r)**2
                expected=.5+.5*r*np.sqrt(1-rho2)/8. if rho2<1 else 0.
                self.assertAlmostEqual(actual,expected,delta=2e-6)
            self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:v.doneCurrent()

    def test_depth_with_intersecting_opaque_plane_and_clip_planes(self):
        v=self.viewer(samples=0);self.require_analytic(v)
        poly=face('p','#00ff00',.3)
        v.set_scene(replace(atoms_scene(),polyhedra=(poly,)))
        v.set_display_options(show_polyhedra=True)
        v.set_polyhedron_opacities({0:1.})
        img=self.pixels(v);w,h=img.shape[1],img.shape[0]
        v.makeCurrent()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject())
            scale=v._screen_scale()*v.devicePixelRatioF()
            for dx in (0,25,48):
                x,y=w//2+dx,h//2
                actual=float(np.asarray(GL.glReadPixels(x,y,1,1,GL.GL_DEPTH_COMPONENT,GL.GL_FLOAT)).ravel()[0])
                rho2=((x+.5-w/2)**2+(y+.5-h/2)**2)/(scale*.5)**2
                expected=.5+.5*max(.3,.5*np.sqrt(1-rho2))/8.
                self.assertAlmostEqual(actual,expected,delta=2e-6)
        finally:v.doneCurrent()
        v.set_display_options(show_polyhedra=False)
        for center_z, expected_z in ((7.8,7.3),(8.8,None),(-8.8,None),(-8.2,-7.7)):
            v.set_scene(atoms_scene(positions=((0.,0.,center_z),)))
            self.pixels(v);v.makeCurrent()
            try:
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject())
                actual=float(np.asarray(GL.glReadPixels(w//2,h//2,1,1,GL.GL_DEPTH_COMPONENT,GL.GL_FLOAT)).ravel()[0])
                expected=0. if expected_z is None else .5+.5*expected_z/8.
                self.assertAlmostEqual(actual,expected,delta=2e-5)
            finally:v.doneCurrent()

    def test_msaa_coverage_and_no_camera_dependent_facets(self):
        v=self.viewer(mixed=True);self.require_analytic(v);v.set_display_options(engraving=True)
        img=self.pixels(v);v.makeCurrent()
        try: self.assertGreater(int(GL.glGetIntegerv(GL.GL_SAMPLES)),1)
        finally:v.doneCurrent()
        rgb=img[:,:,:3];background=rgb[0,0];centre=rgb[img.shape[0]//2,img.shape[1]//2]
        fractional=np.any(rgb!=background,axis=2)&np.any(rgb!=centre,axis=2)
        self.assertGreater(np.count_nonzero(fractional),20)
        with patch.object(v,'_native_gl',None),patch.object(renderer,'glDrawArraysInstanced',wraps=renderer.glDrawArraysInstanced) as draw,patch.object(v,'_rebuild_atom_cache',side_effect=AssertionError('unexpected geometry upload')):
            v.set_orientation(rotation(0,23)@rotation(1,37));np.testing.assert_array_equal(self.pixels(v),img)
            self.assertTrue(any(c.args[:4]==(GL.GL_TRIANGLE_STRIP,0,4,1) for c in draw.call_args_list))
        v.set_display_options(engraving=False);coloured=self.pixels(v)
        v.set_orientation(rotation(2,61));np.testing.assert_array_equal(self.pixels(v),coloured)
        v.set_atom_visibility({0:False});hidden=self.pixels(v)
        np.testing.assert_array_equal(hidden,np.broadcast_to(hidden[0,0],hidden.shape))

    def test_mesh_fallback_for_nonorthogonal_transform_and_shader_failure(self):
        v=self.viewer();self.require_analytic(v);legacy=self.viewer('mesh')
        m=np.array([[1.,.2,0.],[0.,1.,.1],[0.,0.,1.]])
        for a in (v,legacy):a.set_orientation(m)
        np.testing.assert_array_equal(self.pixels(v),self.pixels(legacy));self.assertEqual(v._last_atom_backend,'mesh')
        v.set_orientation(np.eye(3));self.pixels(v);self.assertEqual(v._last_atom_backend,'glsl')
        broken=self.viewer(fail=True);self.assertIn('forced analytic',broken.atom_analytic_error)
        self.assertEqual(broken._atom_analytic_program,0);self.assertEqual(broken._atom_analytic_vao,0)
        legacy.set_orientation(np.eye(3));np.testing.assert_array_equal(self.pixels(broken),self.pixels(legacy))

    def test_cleanup_and_weighted_path(self):
        v=self.viewer(cls=UnitCellViewer);self.require_analytic(v)
        v.set_scene(sample_scene());v.set_display_options(show_polyhedra=True,weighted_blended_oit=True);self.pixels(v)
        self.assertTrue(v._last_frame_wboit);self.assertEqual(v._last_atom_backend,'glsl')
        self.assertIsNone(v.weighted_oit_error)
        vao,program=v._atom_analytic_vao,v._atom_analytic_program
        # Observe deletion at the actual call, before Qt can recycle handles.
        deleted=[];programs=[];dvao=v._resource_gl.glDeleteVertexArrays;dprog=renderer.glDeleteProgram
        def delvao(n,ids):deleted.extend(ids);return dvao(n,ids)
        def delprog(i):programs.append(i);return dprog(i)
        with patch.object(v._resource_gl,'glDeleteVertexArrays',side_effect=delvao),patch.object(renderer,'glDeleteProgram',side_effect=delprog):v.cleanup()
        self.assertIn(vao,deleted);self.assertIn(program,programs)
        self.assertEqual(v._atom_analytic_vao,0);self.assertEqual(v._atom_analytic_program,0)
        with patch.dict(os.environ,{'UNIT_CELL_GUI_ATOM_BACKEND':'glsl'}):self.pixels(v)
        self.assertEqual(v._last_atom_backend,'glsl');self.assertIsNone(v.atom_analytic_error)
