"""Face-coloured contact curves, mode/style changes and partial replacements."""
from dataclasses import replace
import os
import unittest
import numpy as np
import shiboken6
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication
from unit_cell_gui import AtomComponent, StyleOverrides, UnitCellViewer
from test_visibility_cache import repeated_scene
import test_z_updates


class ContactColourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.set_scene(repeated_scene(3))
        self.viewer.set_display_options(show_polyhedra=True)

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)

    def assert_contacts(self, expected):
        records, ranges = self.viewer._build_atom_face_intersection_geometry()
        self.assertEqual({item.poly_index for item in ranges}, set(expected))
        for item in ranges:
            colour = QColor(expected[item.poly_index])
            rgba = np.asarray([colour.redF(), colour.greenF(), colour.blueF(), 1], np.float32)
            block = records[item.first:item.first + item.count]
            np.testing.assert_array_equal(block[:, 6:10], np.tile(rgba, (item.count, 1)))

    def test_each_contact_uses_its_own_face_colour_and_override(self):
        self.assert_contacts({0:'#55aadd', 1:'#55aadd', 2:'#55aadd'})
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_colours={
            'p0':'#00cc44', 'p1':'#cc8800',
        }))
        self.assert_contacts({0:'#00cc44', 1:'#cc8800', 2:'#55aadd'})
        self.viewer.set_style_overrides(StyleOverrides(atom_component_colours={'Nb:2':'#ff0000'}))
        self.assert_contacts({0:'#55aadd', 1:'#55aadd', 2:'#55aadd'})

    def test_mode_and_polyhedron_colours_invalidate_sections_but_opacity_does_not(self):
        self.viewer._intersection_dirty = False
        self.viewer.set_display_options(engraving=True)
        self.assertTrue(self.viewer._intersection_dirty)
        self.assert_contacts({0:'#141414', 1:'#141414', 2:'#141414'})
        self.viewer._intersection_dirty = False
        self.viewer.set_display_options(engraving=False)
        self.assertTrue(self.viewer._intersection_dirty)
        self.assert_contacts({0:'#55aadd', 1:'#55aadd', 2:'#55aadd'})
        self.viewer._intersection_dirty = False
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_colours={'p0':'#ff0088'}))
        self.assertTrue(self.viewer._intersection_dirty)
        self.viewer._intersection_dirty = False
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_colours={'p0':'#ff0088'},
                                                      polyhedron_opaque_sites={'p0'}))
        self.assertFalse(self.viewer._intersection_dirty)

    def test_replaced_mixed_polyhedron_and_reset_use_effective_face_colour(self):
        components = (AtomComponent('r','R','C',.25,'#ff0000',.35),
                      AtomComponent('b','B','O',.75,'#0000ff',.35))
        poly = replace(self.viewer.scene.polyhedra[1], components=components)
        self.viewer.update_polyhedra({1:poly})
        self.assert_contacts({0:'#55aadd', 1:'#4000bf', 2:'#55aadd'})
        self.viewer.reset_scene_updates()
        self.assert_contacts({0:'#55aadd', 1:'#55aadd', 2:'#55aadd'})


class OpenGLContactColourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:
            raise unittest.SkipTest('OpenGL display required')

    def test_mode_style_and_partial_replacement_match_fresh_scene(self):
        # Reuse the existing full-scene reference comparison with a viewer that
        # has already uploaded both base and per-object section buffers.
        case = test_z_updates.OpenGLUpdatesTests(methodName='test_batch_topology_change_and_context_recreation')
        case.app = self.app
        case.setUp()
        try:
            viewer = case.v
            original = case.pixels()
            poly = replace(case.scene.polyhedra[0], components=(
                AtomComponent('green','Green','C',1,'#00cc44',.3),
            ))
            viewer.update_polyhedra({0:poly})
            case.pixels(); case.reference()
            viewer.set_display_options(engraving=True)
            case.pixels(); case.reference()
            viewer.set_display_options(engraving=False)
            viewer.set_style_overrides(StyleOverrides(polyhedron_site_colours={'cube':'#cc8800'}))
            case.pixels(); case.reference()
            recoloured = case.pixels()
            self.assertFalse(np.array_equal(original, recoloured))
            # Compare reset/restoration to its earlier framebuffer as well as
            # the existing exact reference check for a component-colour edit.
            viewer.set_style_overrides(StyleOverrides())
            case.pixels(); case.reference()
            viewer.reset_scene_updates()
            np.testing.assert_array_equal(case.pixels(), original)
            self.assertIsNone(viewer.opengl_error)
        finally:
            case.tearDown()
