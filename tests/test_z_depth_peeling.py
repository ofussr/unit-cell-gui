"""Analytic alpha ordering, hidden edges, many layers and peeling resources."""
import os
import unittest
import numpy as np
import shiboken6
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from OpenGL import GL
from unit_cell_gui import AtomComponent, Polyhedron, Scene, StyleOverrides, UnitCellViewer


def face(key, colour, depth, low=-1, high=1):
    corners = np.array([[-1,low,0], [1,low,0], [1,high,0], [-1,high,0]], float)
    corners[:,2] = depth(corners[:,0]) if callable(depth) else depth
    component = AtomComponent(key, key, 'C', 1, colour, .3)
    return Polyhedron(corners.mean(axis=0)-[0,0,.2], corners, ((0,1,2,3),), key, key, (component,))


def scene(polys):
    return Scene((), [], [], [], [], np.eye(3), tuple(polys), base_radius=2)


class OpenGLDepthPeelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:
            raise unittest.SkipTest('OpenGL display required')

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.resize(640,480)
        self.viewer.set_display_options(show_polyhedra=True, show_atoms=False, show_bonds=False,
                                        show_cell=False, show_basis=False)
        self.viewer.show(); self.app.processEvents()
        self.assertTrue(self.viewer.isValid())
        self.assertIsNone(self.viewer.opengl_error)

    def tearDown(self):
        self.viewer.close(); shiboken6.delete(self.viewer); self.app.processEvents()

    def pixels(self):
        image = self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.viewer.opengl_error)
        data = np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy()
        self.viewer.makeCurrent()
        try:
            self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
            self.assertEqual(int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)),self.viewer.defaultFramebufferObject())
        finally: self.viewer.doneCurrent()
        return data.reshape(image.height(),image.width(),4)

    def sample(self, pixels, x, y):
        point = self.viewer._project_world_point(np.array([x,y,0]))
        ratio = self.viewer.devicePixelRatioF()
        return pixels[round(point.y()*ratio), round(point.x()*ratio), :3].astype(float)

    @staticmethod
    def over(front, back, alpha=.64):
        return alpha*np.asarray(front)+(1-alpha)*np.asarray(back)

    def test_intersecting_faces_have_opposite_pixel_orders_and_ignore_submission_order(self):
        red = face('red','#ff0000',lambda x:.4*x)
        green = face('green','#00ff00',lambda x:-.4*x)
        self.viewer.set_scene(scene((red,green)))
        first = self.pixels()
        background = np.array(self.viewer.palette().base().color().getRgb()[:3])
        left = self.over([0,255,0],self.over([255,0,0],background))
        right = self.over([255,0,0],self.over([0,255,0],background))
        np.testing.assert_allclose(self.sample(first,-.5,.25),left,atol=1)
        np.testing.assert_allclose(self.sample(first,.5,.25),right,atol=1)
        self.viewer.set_scene(scene((green,red)),reset_camera=False)
        second = self.pixels()
        np.testing.assert_array_equal(first,second)

    def test_back_edge_is_attenuated_by_front_face_at_its_actual_depth(self):
        front = face('purple','#a020f0',.4)
        behind = face('green','#008800',-.4,low=0)
        self.viewer.set_scene(scene((front,behind)))
        pixels = self.pixels()
        # At the green lower edge, everything behind the purple front face
        # contributes at most 1-alpha, even if it is fully saturated green.
        actual = self.sample(pixels,0,0)
        purple = np.array([160,32,240])
        self.assertLessEqual(actual[1],self.over(purple,[255,255,255])[1]+1)
        self.assertGreaterEqual(actual[0],.64*purple[0]-1)
        self.assertGreaterEqual(actual[2],.64*purple[2]-1)
        self.assertGreater(self.viewer._depth_peeler.layers,1)

    def test_all_layers_are_processed_without_fixed_eight_or_sixteen_layer_cutoff(self):
        polygons = [face(str(i),'#4488cc',-.8+i*.04) for i in range(40)]
        self.viewer.set_scene(scene(polygons))
        pixels = self.pixels()
        self.assertGreaterEqual(self.viewer._depth_peeler.layers,40)
        np.testing.assert_allclose(self.sample(pixels,0,.25),[68,136,204],atol=1)

    def test_opaque_front_face_blocks_deeper_fragments(self):
        self.viewer.set_scene(scene((face('front','#008800',.4),face('back','#ff0000',-.4))))
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_opaque_sites={'front'}))
        pixels = self.pixels()
        np.testing.assert_array_equal(self.sample(pixels,0,.25),[0,136,0])

    def test_targets_reuse_resize_release_on_hidden_scene_and_reconstruct_after_cleanup(self):
        self.viewer.set_scene(scene((face('purple','#a020f0',.4),face('green','#008800',-.4,low=0))))
        original = self.pixels()
        peeler = self.viewer._depth_peeler
        targets = [target.copy() for target in peeler.targets]
        self.pixels(); self.assertEqual(peeler.targets,targets)
        self.assertEqual(peeler.size,(original.shape[1]*2,original.shape[0]*2))
        self.viewer.resize(760,520); self.app.processEvents(); self.pixels()
        self.assertEqual(peeler.size,(round(760*self.viewer.devicePixelRatioF())*2,
                                      round(520*self.viewer.devicePixelRatioF())*2))
        self.viewer.makeCurrent()
        try:
            GL.glBindTexture(GL.GL_TEXTURE_2D,peeler.base['colour'])
            self.assertEqual(int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D,0,GL.GL_TEXTURE_WIDTH)),peeler.size[0])
            GL.glBindTexture(GL.GL_TEXTURE_2D,0)
        finally: self.viewer.doneCurrent()
        self.viewer.cleanup(); self.viewer.makeCurrent()
        try:
            self.viewer.initializeGL()
        finally: self.viewer.doneCurrent()
        self.pixels()
        new = self.viewer._depth_peeler
        textures = [target[key] for target in new.targets for key in ('colour','depth') if target[key]]
        framebuffers = [target['fbo'] for target in new.targets]
        self.viewer.set_display_options(show_polyhedra=False); self.pixels()
        self.assertFalse(new.targets)
        self.viewer.makeCurrent()
        try:
            self.assertTrue(all(not GL.glIsTexture(value) for value in textures))
            self.assertTrue(all(not GL.glIsFramebuffer(value) for value in framebuffers))
        finally: self.viewer.doneCurrent()
        self.viewer.set_display_options(show_polyhedra=True); self.pixels()
        self.viewer.set_scene(None); self.pixels()
        self.assertFalse(new.targets)
