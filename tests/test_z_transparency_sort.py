"""Geometric face/edge order, native MSAA and retained partial updates."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from OpenGL import GL
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import DisplayOptions, StyleOverrides, UnitCellViewer
from unit_cell_gui._transparency_sort import back_to_front
from unit_cell_gui import opengl_viewer as renderer
from unit_cell_gui import _transparency_sort as sorter
from render_test_helpers import face,scene,adjacent_cubes
from test_unit_cell_gui import rotation,sample_scene


class GeometricOrderTests(unittest.TestCase):
    def test_spatial_candidates_match_exhaustive_bounds_checks(self):
        rng = np.random.default_rng(2572)
        low = rng.uniform(-10, 10, (180, 2))
        boxes = np.column_stack((low, low + rng.uniform(0, 3, (180, 2))))
        # Touching, coincident, zero-sized and just-separated boxes.
        boxes = np.vstack((boxes, [0,0,1,1], [1,1,2,2], [0,0,1,1],
                           [2,2,2,2], [2+5e-10,2,3,3]))
        for epsilon in (0., 1e-9):
            pairs = [tuple(pair) for batch in sorter._overlapping_pair_batches(boxes, epsilon)
                     for pair in batch]
            expected = {(i,j) for i,a in enumerate(boxes) for j,b in enumerate(boxes)
                        if i < j and a[0] <= b[2]+epsilon and b[0] <= a[2]+epsilon
                        and a[1] <= b[3]+epsilon and b[1] <= a[3]+epsilon}
            self.assertEqual(set(pairs), expected)
            self.assertEqual(len(pairs), len(expected))
        self.assertEqual(list(sorter._overlapping_pair_batches(np.empty((0,4)),0.)), [])

    def test_spatial_order_matches_exhaustive_candidate_order(self):
        current = adjacent_cubes(12)
        primitives = []
        for poly in current.polyhedra:
            for indices in poly.faces:
                points = poly.vertices[list(indices)]
                primitives.append((False, points, (poly.site_key,)))
                for p,q in zip(points, np.roll(points,-1,axis=0)):
                    primitives.append((True, np.array([p,q]), (poly.site_key,)))
        def exhaustive(boxes, epsilon):
            pairs = [(i,j) for i,a in enumerate(boxes) for j,b in enumerate(boxes)
                     if i < j and a[0] <= b[2]+epsilon and b[0] <= a[2]+epsilon
                     and a[1] <= b[3]+epsilon and b[1] <= a[3]+epsilon]
            if pairs:
                yield np.asarray(pairs)
        for orientation in (np.eye(3), rotation(0,23)@rotation(1,37), rotation(2,71)):
            actual = back_to_front(primitives, orientation, line_margin=.002)
            with patch.object(sorter, '_overlapping_pair_batches', exhaustive):
                expected = back_to_front(primitives, orientation, line_margin=.002)
            self.assertEqual(actual, expected)

    def test_sparse_scene_does_not_run_narrow_phase_on_distant_objects(self):
        primitives = [(False, face(str(i),'#4488cc',0).vertices+[i*4.,0,0], (str(i),))
                      for i in range(1000)]
        with patch.object(sorter, '_differences', wraps=sorter._differences) as exact:
            order, conflicts = back_to_front(primitives, np.eye(3))
        exact.assert_not_called()
        self.assertEqual(len(order),1000)
        self.assertEqual(conflicts,0)

    def test_overlap_depth_overrides_wrong_centroid_order(self):
        large=face('red','#ff0000',lambda x:.5*x)
        small=replace(face('green','#00ff00',.3),vertices=np.array([
            [.8,-.5,.3],[1.2,-.5,.3],[1.2,.5,.3],[.8,.5,.3]]))
        order,conflicts=back_to_front([(False,large.vertices,('r',)),(False,small.vertices,('g',))],np.eye(3))
        self.assertEqual(order,[1,0]);self.assertEqual(conflicts,0)

    def test_edge_behind_face_is_drawn_before_it(self):
        polygon=face('p','#ff0000',.4).vertices
        edge=np.array([[-.7,0,-.4],[.7,0,-.4]])
        order,conflicts=back_to_front([(False,polygon,('p',)),(True,edge,('e',))],np.eye(3))
        self.assertEqual(order,[1,0]);self.assertEqual(conflicts,0)

    def test_coplanar_order_uses_stable_material_key(self):
        points=face('p','#ff0000',0).vertices
        primitives=[(False,points,('red',)),(False,points,('green',))]
        first,_=back_to_front(primitives,np.eye(3))
        second,_=back_to_front(primitives[::-1],np.eye(3))
        self.assertEqual([primitives[i][2] for i in first],[primitives[::-1][i][2] for i in second])

    def test_intersections_are_diagnostic_not_a_hidden_peeling_fallback(self):
        first=face('r','#ff0000',lambda x:.4*x)
        second=face('g','#00ff00',lambda x:-.4*x)
        order,conflicts=back_to_front([(False,first.vertices,('r',)),(False,second.vertices,('g',))],np.eye(3))
        self.assertEqual(sorted(order),[0,1]);self.assertGreater(conflicts,0)

    def test_legacy_supersampling_setting_is_still_accepted(self):
        self.assertEqual(DisplayOptions(transparency_supersampling=2).transparency_supersampling,2)
        for value in (True,0,3,1.,'2'):
            with self.assertRaises(ValueError):DisplayOptions(transparency_supersampling=value)


class OpenGLGlobalOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('OpenGL display required')
    def setUp(self):
        self.viewer=UnitCellViewer();self.viewer.resize(640,480)
        self.viewer.set_display_options(show_polyhedra=True,show_atoms=False,show_bonds=False,
                                       show_cell=False,show_basis=False)
        self.viewer.show();self.app.processEvents()
    def tearDown(self):
        self.viewer.close();shiboken6.delete(self.viewer);self.app.processEvents()
    def pixels(self):
        image=self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.viewer.opengl_error)
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)
    def sample(self,pixels,x=0,y=.25):
        p=self.viewer._project_world_point(np.array([x,y,0]));ratio=self.viewer.devicePixelRatioF()
        return pixels[round(p.y()*ratio),round(p.x()*ratio),:3].astype(float)

    def test_nonintersecting_slanted_faces_ignore_submission_and_centroid_order(self):
        red=face('red','#ff0000',lambda x:.5*x)
        green=replace(face('green','#00ff00',.3),vertices=np.array([
            [.8,-.5,.3],[1.2,-.5,.3],[1.2,.5,.3],[.8,.5,.3]]))
        self.viewer.set_scene(scene((red,green)));first=self.pixels()
        background=np.array(self.viewer.palette().base().color().getRgb()[:3])
        expected=.64*np.array([255,0,0])+.36*(.64*np.array([0,255,0])+.36*background)
        np.testing.assert_allclose(self.sample(first,.9,.1),expected,atol=1)
        self.viewer.set_scene(scene((green,red)),reset_camera=False)
        np.testing.assert_array_equal(self.pixels(),first)

    def test_rear_edge_is_attenuated_by_front_face(self):
        self.viewer.set_scene(scene((face('purple','#a020f0',.4),face('green','#008800',-.4,low=0))))
        pixels=self.pixels();actual=self.sample(pixels,0,0)
        purple=np.array([160,32,240])
        self.assertLessEqual(actual[1],(.64*purple+.36*255)[1]+1)
        self.assertGreaterEqual(actual[0],.64*purple[0]-1)
        self.assertEqual(self.viewer._transparency_sort_conflicts,0)

    def test_all_forty_faces_are_drawn_and_msaa_is_retained_without_peeling_targets(self):
        self.viewer.set_scene(scene(tuple(face(str(i),'#4488cc',-.8+i*.04) for i in range(40))))
        with patch.object(GL,'glTexImage2D') as texture,patch.object(GL,'glGenFramebuffers') as fbo, \
             patch.object(GL,'glBeginQuery') as query:
            pixels=self.pixels()
            texture.assert_not_called();fbo.assert_not_called();query.assert_not_called()
        np.testing.assert_allclose(self.sample(pixels),[68,136,204],atol=1)
        self.assertEqual(sum(not edge for edge,_ in self.viewer._transparent_order),40)
        self.viewer.makeCurrent()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.viewer.defaultFramebufferObject())
            self.assertGreater(int(GL.glGetIntegerv(GL.GL_SAMPLES)),1)
            self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:self.viewer.doneCurrent()

    def test_legacy_quality_flag_is_noop_and_context_recreation_preserves_frame(self):
        self.viewer.set_scene(scene((face('p','#4488cc',.3),face('q','#cc8844',-.3))))
        self.viewer.set_polyhedron_opacities({0:.35});original=self.pixels()
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate:
            self.viewer.set_display_options(transparency_supersampling=2)
            np.testing.assert_array_equal(self.pixels(),original)
            allocate.assert_not_called()
        self.viewer.cleanup();self.viewer.makeCurrent()
        try:self.viewer.initializeGL()
        finally:self.viewer.doneCurrent()
        np.testing.assert_array_equal(self.pixels(),original)

    def test_close_show_restores_scene_and_updates_without_manual_initialization(self):
        self.viewer.set_display_options(show_atoms=True,show_bonds=True)
        self.viewer.set_scene(sample_scene())
        self.viewer.set_orientation(rotation(0,23)@rotation(1,37))
        positions = self.viewer.get_atom_positions()
        self.viewer.set_atom_positions({0:positions[0]+[.07,.04,0]})
        self.viewer.set_polyhedron_opacities({0:.35})
        for engraving in (False,True):
            self.viewer.set_display_options(engraving=engraving)
            original = self.pixels()
            expected_opacity = self.viewer.get_polyhedron_opacities()[0]
            self.assertGreater(len(np.unique(original.reshape(-1,4),axis=0)),1)
            for _ in range(2):
                self.viewer.close(); self.app.processEvents()
                self.assertEqual(self.viewer._program,0)
                # Exercise Qt's normal lifecycle: no initializeGL/makeCurrent.
                self.viewer.show(); self.app.processEvents()
                np.testing.assert_array_equal(self.pixels(),original)
                self.assertNotEqual(self.viewer._program,0)
                self.assertAlmostEqual(self.viewer.get_polyhedron_opacities()[0],expected_opacity)

    def test_camera_keeps_geometry_resident_and_pan_reuses_visibility_order(self):
        self.viewer.set_scene(scene((face('p','#4488cc',.3),face('q','#cc8844',-.3))))
        self.pixels()
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate, \
             patch.object(renderer,'glBufferSubData',wraps=renderer.glBufferSubData) as upload, \
             patch.object(renderer,'back_to_front',wraps=renderer.back_to_front) as order:
            self.viewer.pan_x=20;self.viewer.update();self.pixels()
            order.assert_not_called()
            self.viewer.set_orientation(np.array([[.8,0,.6],[0,1,0],[-.6,0,.8]]));self.pixels()
            order.assert_called_once();allocate.assert_not_called();upload.assert_not_called()

    def test_shared_faces_and_edges_keep_per_object_materials_and_ignore_input_order(self):
        original=adjacent_cubes(6)
        for orientation in (np.eye(3),rotation(1,90),rotation(0,23)@rotation(1,37)):
            self.viewer.set_scene(original)
            self.viewer.set_orientation(orientation)
            first=self.pixels()
            self.assertEqual(self.viewer._transparency_sort_conflicts,0)
            self.viewer.set_scene(replace(original,polyhedra=original.polyhedra[::-1]),reset_camera=False)
            np.testing.assert_array_equal(self.pixels(),first)

    def test_atoms_and_bonds_keep_identical_smoothing_when_transparency_is_enabled(self):
        original=replace(sample_scene(),polyhedra=())
        self.viewer.set_display_options(show_atoms=True,show_bonds=True)
        self.viewer.set_scene(original)
        self.viewer.set_orientation(rotation(0,23)@rotation(2,31))
        reference=self.pixels()
        offscreen=face('offscreen','#22bb55',0)
        offscreen=replace(offscreen,center=offscreen.center+[1000,0,0],
                          vertices=offscreen.vertices+[1000,0,0])
        self.viewer.set_scene(replace(original,polyhedra=(offscreen,)),reset_camera=False)
        np.testing.assert_array_equal(self.pixels(),reference)
