"""Atomic opacity packets, four-byte updates and render-pass transitions."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from OpenGL import GL
from unit_cell_gui import StyleOverrides, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from test_z_depth_peeling import face, scene
import test_z_updates


class OpacityApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.viewer=UnitCellViewer()
        self.scene=scene((face('red','#ff0000',.3),face('green','#00ff00',-.3)))
        self.viewer.set_scene(self.scene)
    def tearDown(self): self.viewer.close(); shiboken6.delete(self.viewer)

    def test_invalid_packet_is_atomic_and_no_scene_getter_is_empty(self):
        for value,kind in [(-.1,ValueError),(1.1,ValueError),(float('nan'),ValueError),
                           (float('inf'),ValueError),(True,TypeError),('0.4',TypeError),([.4],TypeError)]:
            with self.assertRaises(kind):
                self.viewer.update_scene(polyhedra={0:replace(self.scene.polyhedra[0],vertices=self.scene.polyhedra[0].vertices*1.1)},
                                         polyhedron_opacities={1:value})
            self.assertIsNone(self.viewer._scene_updates)
        with self.assertRaises(IndexError): self.viewer.set_polyhedron_opacities({2:.4})
        self.viewer.set_scene(None)
        self.assertEqual(self.viewer.get_polyhedron_opacities().shape,(0,))
        with self.assertRaises(RuntimeError): self.viewer.set_polyhedron_opacities({0:.4})

    def test_merge_defensive_getter_defaults_styles_and_reset(self):
        values={0:.35,1:1}
        with patch.object(self.viewer,'update') as repaint:
            self.viewer.set_polyhedron_opacities(values); repaint.assert_called_once()
        values[0]=.8
        np.testing.assert_array_equal(self.viewer.get_polyhedron_opacities(),[.35,1])
        copy=self.viewer.get_polyhedron_opacities(); copy[:]=0
        self.assertEqual(self.viewer.get_polyhedron_opacities()[0],.35)
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_opaque_sites={'red'}))
        self.assertEqual(self.viewer.get_polyhedron_opacities()[0],.35)
        self.viewer.set_polyhedron_opacities({0:None})
        np.testing.assert_array_equal(self.viewer.get_polyhedron_opacities(),[1,1])
        self.viewer.reset_scene_updates()
        np.testing.assert_array_equal(self.viewer.get_polyhedron_opacities(),[1,.64])
        self.assertIs(self.viewer.scene,self.scene)
        self.viewer.set_polyhedron_opacities({0:.25,1:0})
        self.viewer.set_display_options(engraving=True)
        np.testing.assert_array_equal(self.viewer.get_polyhedron_opacities(),[1,0])
        self.viewer.set_display_options(engraving=False)
        np.testing.assert_array_equal(self.viewer.get_polyhedron_opacities(),[.25,0])
        with patch.object(self.viewer,'update') as repaint:
            self.viewer.set_polyhedron_opacities({0:.25}); repaint.assert_not_called()


class OpenGLOpacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}: raise unittest.SkipTest('OpenGL display required')
    def setUp(self):
        self.viewer=UnitCellViewer(); self.viewer.resize(640,480)
        self.scene=scene((face('red','#ff0000',.3),face('green','#00ff00',-.3)))
        self.viewer.set_scene(self.scene)
        self.viewer.set_display_options(show_polyhedra=True,show_cell=False,show_basis=False)
        self.viewer.show(); self.app.processEvents(); self.pixels()
    def tearDown(self): self.viewer.close(); shiboken6.delete(self.viewer); self.app.processEvents()
    def pixels(self):
        im=self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.viewer.opengl_error)
        return np.frombuffer(im.bits(),np.uint8,count=im.sizeInBytes()).copy().reshape(im.height(),im.width(),4)
    def centre(self,pixels): return pixels[pixels.shape[0]//2,pixels.shape[1]//2,:3]

    def test_single_and_adjacent_opacity_updates_only_upload_material_floats(self):
        original=self.pixels()
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate, \
             patch.object(renderer,'glBufferSubData',wraps=renderer.glBufferSubData) as upload, \
             patch.object(self.viewer,'_build_static_geometry',wraps=self.viewer._build_static_geometry) as geometry:
            self.viewer.set_polyhedron_opacities({0:.25}); self.pixels()
            allocate.assert_not_called(); geometry.assert_not_called()
            self.assertEqual([call.args[2] for call in upload.call_args_list],[4])
            upload.reset_mock()
            self.viewer.set_polyhedron_opacities({0:.4,1:.5}); self.pixels()
            self.assertEqual([call.args[2] for call in upload.call_args_list],[8])
        self.viewer.reset_scene_updates()
        np.testing.assert_array_equal(self.pixels(),original)

    def test_zero_fractional_and_opaque_switches_match_independent_alpha_formula(self):
        background=np.array(self.viewer.palette().base().color().getRgb()[:3],float)
        red=np.array([255,0,0]);green=np.array([0,255,0])
        for front in [0,.25,1,.8,0]:
            with patch.object(self.viewer,'_build_static_geometry',wraps=self.viewer._build_static_geometry) as geometry:
                self.viewer.set_polyhedron_opacities({0:front,1:.5})
                pixels=self.pixels(); geometry.assert_not_called()
            expected=front*red+(1-front)*(.5*green+.5*background)
            np.testing.assert_allclose(self.centre(pixels),expected,atol=1)
            if front==1: self.assertTrue(any(r.poly_index==0 for r in self.viewer._effective_ranges('opaque')))
            if front==0: self.assertFalse(any(r.poly_index==0 for r in self.viewer._visible_polyhedron_ranges(self.viewer._effective_ranges('face_edge'))))

    def test_native_opaque_buffer_can_become_transparent_without_reallocation(self):
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_opaque_sites={'red','green'})); self.pixels()
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate, \
             patch.object(self.viewer,'_build_static_geometry',wraps=self.viewer._build_static_geometry) as geometry:
            self.viewer.set_polyhedron_opacities({0:.25,1:.5}); pixels=self.pixels()
            allocate.assert_not_called(); geometry.assert_not_called()
        expected=.25*np.array([255,0,0])+.75*(.5*np.array([0,255,0])+.5*np.array([255,255,255]))
        np.testing.assert_allclose(self.centre(pixels),expected,atol=1)

    def test_opacity_survives_context_recreation_and_set_scene_clears_it(self):
        self.viewer.set_polyhedron_opacities({0:.25,1:.5}); before=self.pixels()
        vbo=self.viewer._poly_opacity_vbo; texture=self.viewer._poly_opacity_texture
        self.viewer.cleanup(); self.viewer.makeCurrent()
        try:
            self.assertFalse(GL.glIsBuffer(vbo)); self.assertFalse(GL.glIsTexture(texture))
            self.viewer.initializeGL()
        finally: self.viewer.doneCurrent()
        np.testing.assert_array_equal(self.pixels(),before)
        self.viewer.set_scene(self.scene,reset_camera=False); self.pixels()
        np.testing.assert_array_equal(self.viewer.get_polyhedron_opacities(),[.64,.64])

    def test_batch_replacement_atoms_opacity_and_engraving_zero_match_full_reference(self):
        case=test_z_updates.OpenGLUpdatesTests(methodName='test_batch_topology_change_and_context_recreation'); case.app=self.app
        case.setUp()
        try:
            v=case.v
            v.update_scene(atom_positions={1:[.6,.2,.1]},polyhedra={0:replace(case.scene.polyhedra[0],vertices=case.scene.polyhedra[0].vertices*1.1)},
                           polyhedron_opacities={0:.3,1:0})
            case.pixels();case.reference()
            v.set_display_options(engraving=True);case.pixels();case.reference()
            self.assertNotIn(1,v._hatch_draw_indices)
            v.reset_scene_updates();case.pixels();case.reference()
        finally: case.tearDown()
