"""External partial updates: atomic validation, immutable input and GL locality."""
import os
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
import shiboken6
from unit_cell_gui import Atom, AtomComponent, Polyhedron, Scene, StyleOverrides, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from test_z_scene_instancing import cube_scene
from test_unit_cell_gui import rotation


def separated_scene():
    base=cube_scene(); second=base.polyhedra[1]
    shift=np.array([4.2,-.3,.4])
    second=replace(second, center=second.center+shift, vertices=second.vertices+shift)
    c=AtomComponent('C','C','C',1,'#88aa55',.3)
    return Scene(base.atoms+(Atom('x','X',(c,)),Atom('y','Y',(c,))),
                 np.vstack((base.atom_centers,[[4,0,0],[4.7,0,0]])),[[0,1],[2,3]],
                 base.cell_vertices,base.cell_edges,base.basis_vectors,(base.polyhedra[0],second),6.)


class UpdateApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.v=UnitCellViewer(); self.scene=separated_scene(); self.v.set_scene(self.scene)
    def tearDown(self): self.v.close(); shiboken6.delete(self.v)

    def test_atomic_batch_copies_data_and_preserves_scene_and_camera(self):
        before=self.scene.atom_centers.copy(); self.v.zoom=1.7; self.v.pan_x=23
        matrix=rotation(1,31); self.v.set_orientation(matrix)
        point=np.array([.1,.2,.3]); poly=replace(self.scene.polyhedra[0],vertices=self.scene.polyhedra[0].vertices*1.1)
        with patch.object(self.v,'update') as repaint:
            self.v.update_scene(atom_positions={0:point},atom_visibility={1:False},polyhedra={0:poly})
            repaint.assert_called_once()
        point[:]=99
        self.assertIs(self.v.scene,self.scene)
        np.testing.assert_array_equal(self.scene.atom_centers,before)
        np.testing.assert_array_equal(self.v.get_atom_positions()[0],[.1,.2,.3])
        self.assertEqual(self.v.get_atom_visibility().tolist(),[True,False,True,True])
        self.assertEqual(self.v.zoom,1.7); self.assertEqual(self.v.pan_x,23)
        np.testing.assert_array_equal(self.v.orientation,matrix)
        result=self.v.get_atom_positions(); result[:]=100
        self.assertLess(self.v.get_atom_positions().max(),100)
        self.assertEqual(self.v._pending_bonds,{0}); self.assertEqual(self.v._pending_polyhedra,{0})
        self.assertNotIn(1,self.v._pending_sections)

    def test_invalid_packets_never_commit_any_part(self):
        for kwargs,kind in ((dict(atom_positions={0:[.1,.2,.3]},atom_visibility={9:False}),IndexError),
                            (dict(atom_positions={0:[float('nan'),0,0]}),ValueError),
                            (dict(atom_positions={0:[1e100,0,0]}),ValueError),
                            (dict(atom_visibility={True:False}),TypeError),
                            (dict(atom_visibility={0:'false'}),TypeError),
                            (dict(polyhedra={0:'poly'}),TypeError),
                            (dict(polyhedra={-1:self.scene.polyhedra[0]}),IndexError),
                            (dict(atom_positions=[(0,[1,2,3])]),TypeError)):
            with self.assertRaises(kind): self.v.update_scene(**kwargs)
            self.assertIsNone(self.v._scene_updates)
            np.testing.assert_array_equal(self.v.get_atom_positions(),self.scene.atom_centers)

    def test_merge_reset_and_full_set_scene_contract(self):
        self.v.set_atom_positions({0:[.1,0,0]}); self.v.set_atom_positions({1:[1.1,0,0]})
        np.testing.assert_array_equal(self.v.get_atom_positions()[:2],[[.1,0,0],[1.1,0,0]])
        self.v.set_atom_visibility({1:False})
        self.v.set_style_overrides(StyleOverrides(atom_component_visibility={'C':False}))
        self.v.set_atom_visibility({1:True})
        self.assertEqual(self.v.get_atom_visibility().tolist(),[True,True,False,False])
        self.v.reset_scene_updates()
        np.testing.assert_array_equal(self.v.get_atom_positions(),self.scene.atom_centers)
        self.assertEqual(self.v.get_atom_visibility().tolist(),[True,True,False,False])
        self.v.set_atom_positions({0:[.3,0,0]}); self.v.set_scene(self.scene,reset_camera=False)
        np.testing.assert_array_equal(self.v.get_atom_positions(),self.scene.atom_centers)
        self.v.set_scene(None)
        self.assertEqual(self.v.get_atom_positions().shape,(0,3))
        with self.assertRaises(RuntimeError): self.v.set_atom_visibility({0:False})

    def test_noop_does_not_request_a_frame(self):
        with patch.object(self.v,'update') as repaint:
            self.v.update_scene(atom_positions={0:self.scene.atom_centers[0]},atom_visibility={0:True},
                                polyhedra={0:replace(self.scene.polyhedra[0])})
            repaint.assert_not_called()


class OpenGLUpdatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}: raise unittest.SkipTest('OpenGL display required')
    def setUp(self):
        self.v=UnitCellViewer(); self.scene=separated_scene(); self.v.resize(640,480)
        self.v.set_scene(self.scene); self.v.set_display_options(show_polyhedra=True)
        self.v.set_orientation(rotation(0,19)@rotation(1,31)); self.v.show(); self.app.processEvents()
        self.assertTrue(self.v.isValid()); self.assertIsNone(self.v.opengl_error)
        self.pixels()
    def tearDown(self): self.v.close(); shiboken6.delete(self.v); self.app.processEvents()
    def pixels(self, viewer=None):
        viewer=viewer or self.v
        im=viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(viewer.opengl_error)
        return np.frombuffer(im.bits(),np.uint8,count=im.sizeInBytes()).copy()
    def reference(self):
        centers=self.v.get_atom_positions(); visible=self.v.get_atom_visibility()
        centers[~visible]=1e6
        bonds=[pair for pair in self.scene.bonds if visible[pair[0]] and visible[pair[1]]]
        scene=Scene(self.scene.atoms,centers,bonds,self.scene.cell_vertices,self.scene.cell_edges,
                    self.scene.basis_vectors,self.v.get_polyhedra(),self.scene.base_radius,self.scene.elements)
        other=UnitCellViewer(); other.resize(self.v.size()); other.set_scene(scene)
        other.set_display_options(show_polyhedra=True,engraving=self.v.engraving,atom_scale=self.v.atom_scale)
        other.set_style_overrides(StyleOverrides(atom_component_visibility=self.v.atom_component_visibility,
            polyhedron_site_visibility=self.v.polyhedron_site_visibility,
            polyhedron_site_colours=self.v.polyhedron_site_colours,
            polyhedron_opaque_sites=self.v.polyhedron_opaque_sites))
        other.set_orientation(self.v.orientation); other.zoom=self.v.zoom
        # Hiding is also reflected by the legend; moving offscreen alone is
        # no longer an equivalent reference for an externally hidden atom.
        other.set_atom_visibility({i:False for i,value in enumerate(visible) if not value})
        if self.v._scene_updates and self.v._scene_updates.opacity_overrides:
            other.set_polyhedron_opacities(self.v._scene_updates.opacity_overrides)
        other.pan_x=self.v.pan_x
        other.pan_y=self.v.pan_y
        other.show(); self.app.processEvents()
        try: np.testing.assert_array_equal(self.pixels(),self.pixels(other))
        finally: other.close(); shiboken6.delete(other); self.app.processEvents()

    def test_one_atom_patches_only_its_rows_and_connected_bond(self):
        before=self.pixels(); untouched=self.v._bond_instance_data[2:].copy()
        with patch.object(self.v,'_rebuild_gpu_cache',wraps=self.v._rebuild_gpu_cache) as all_polys, \
             patch.object(self.v,'_build_atom_instances',wraps=self.v._build_atom_instances) as all_atoms, \
             patch.object(self.v,'_build_colour_bond_instances',wraps=self.v._build_colour_bond_instances) as bonds, \
             patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate:
            self.v.set_atom_positions({1:[.6,.2,.1]}); self.pixels()
            all_polys.assert_not_called(); all_atoms.assert_not_called()
            self.assertEqual(bonds.call_args.kwargs['bond_indices'],[0])
            # Only the affected polyhedron's variable-length section buffer may grow.
            self.assertTrue(all(call.args[1] < 50000 for call in allocate.call_args_list))
        np.testing.assert_array_equal(self.v._bond_instance_data[2:],untouched)
        self.assertFalse(np.array_equal(self.pixels(),before)); self.reference()
        self.v.reset_scene_updates(); np.testing.assert_array_equal(self.pixels(),before)

    def test_hide_move_and_restore_with_host_visibility_and_bw(self):
        before=self.pixels()
        self.v.update_scene(atom_positions={1:[.6,.2,.1]},atom_visibility={1:False})
        self.pixels(); self.reference()
        self.v.set_display_options(engraving=True); self.pixels(); self.reference()
        self.v.set_atom_visibility({1:True}); self.pixels(); self.reference()
        self.v.set_style_overrides(StyleOverrides(atom_component_visibility={'C':False}))
        self.pixels(); self.v.reset_scene_updates(); self.pixels()
        self.assertEqual(self.v.get_atom_visibility().tolist(),[True,True,False,False])
        self.v.set_style_overrides(StyleOverrides()); self.v.set_display_options(engraving=False)
        np.testing.assert_array_equal(self.pixels(),before)

    def test_polyhedron_replacement_reuses_other_geometry_and_hatching(self):
        self.v.set_display_options(engraving=True); before=self.pixels()
        hatch=self.v._hatch_buffers[1]; key=hatch.geometry_key
        poly=replace(self.scene.polyhedra[0],vertices=self.scene.polyhedra[0].vertices*1.12)
        with patch.object(self.v,'_build_static_geometry',wraps=self.v._build_static_geometry) as build, \
             patch.object(self.v,'_build_polyhedron_hatch_geometry',wraps=self.v._build_polyhedron_hatch_geometry) as stroke:
            self.v.update_polyhedra({0:poly}); self.pixels()
            self.assertEqual(build.call_args.kwargs,dict(poly_indices=(0,),include_cell=False))
            self.assertTrue(all(call.args[0]==0 for call in stroke.call_args_list))
        self.assertIs(self.v._hatch_buffers[1],hatch); self.assertEqual(hatch.geometry_key,key)
        self.reference()
        self.v.reset_scene_updates(); np.testing.assert_array_equal(self.pixels(),before)

    def test_batch_topology_change_and_context_recreation(self):
        old=self.scene.polyhedra[0]
        tetra=replace(old,vertices=[[0,0,1],[1,0,-.5],[-.5,.8,-.5],[-.5,-.8,-.5]],
                      faces=((0,1,2),(0,2,3),(0,3,1),(1,3,2)))
        self.v.update_scene(atom_positions={1:[.6,.2,.1]},atom_visibility={2:False},polyhedra={0:tetra})
        self.pixels(); self.reference()
        updated=self.pixels(); self.v.cleanup(); self.v.makeCurrent()
        try: self.v.initializeGL()
        finally: self.v.doneCurrent()
        np.testing.assert_array_equal(self.pixels(),updated)
        self.v.reset_scene_updates(); self.pixels(); self.reference()

    def test_updates_before_first_frame_and_style_refresh_keep_overrides(self):
        self.v.set_scene(self.scene,reset_camera=False)
        poly=replace(self.scene.polyhedra[0],vertices=self.scene.polyhedra[0].vertices*1.1)
        self.v.update_scene(atom_positions={1:[.6,.2,.1]},atom_visibility={2:False},polyhedra={0:poly})
        self.pixels(); self.reference()
        self.v.set_style_overrides(StyleOverrides(polyhedron_opaque_sites={'cube'}))
        self.pixels(); self.reference()
        self.assertFalse(self.v.get_atom_visibility()[2])
        np.testing.assert_array_equal(self.v.get_atom_positions()[1],[.6,.2,.1])
        self.v.reset_scene_updates(); self.pixels(); self.reference()

    def test_repeated_poly_edits_use_subdata_and_scene_clear_releases_resources(self):
        from OpenGL.GL import glIsTexture,glIsBuffer
        poly=replace(self.scene.polyhedra[0],vertices=self.scene.polyhedra[0].vertices*1.1)
        self.v._primitive_texel_capacity=8
        self.v.update_polyhedra({0:poly}); self.pixels(); self.reference()
        prefixes=self.v._partial_poly_gpu[0]['prefixes'][:]
        buffers=[getattr(self.v,prefix+'_vbo') for prefix in prefixes]
        textures=[getattr(self.v,prefix+'_texture') for prefix in prefixes]
        with patch.object(self.v,'_upload_primitive',wraps=self.v._upload_primitive) as allocate:
            self.v.update_polyhedra({0:replace(poly,vertices=poly.vertices*1.03)})
            self.pixels(); allocate.assert_not_called()
        self.reference()
        self.v.set_scene(None); self.pixels(); self.v.makeCurrent()
        try:
            self.assertTrue(all(not glIsBuffer(x) for x in buffers))
            self.assertTrue(all(not glIsTexture(x) for x in textures))
        finally: self.v.doneCurrent()
        self.v.set_scene(self.scene); self.pixels(); self.reference()

    def test_pending_hidden_layer_sections_refresh_when_enabled(self):
        self.v.set_display_options(show_polyhedra=False)
        self.v.set_atom_positions({1:[.6,.2,.1]}); self.pixels()
        self.assertTrue(self.v._pending_sections)
        self.v.set_display_options(show_polyhedra=True); self.pixels()
        self.assertFalse(self.v._pending_sections); self.reference()

    def test_zero_length_bond_can_become_visible_without_reallocation(self):
        self.v.set_atom_positions({1:self.scene.atom_centers[0]}); self.pixels()
        np.testing.assert_array_equal(self.v._bond_instance_data[:2,3:9],np.zeros((2,6)))
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate:
            self.v.set_atom_positions({1:[1,0,0]}); self.pixels()
        self.reference()
        self.assertGreater(np.linalg.norm(self.v._bond_instance_data[0,3:9]),0)
