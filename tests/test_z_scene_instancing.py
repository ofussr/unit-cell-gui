"""Cylinder fidelity, intact face topology and resident primitive instances."""
import os
import unittest
from unittest.mock import patch
import numpy as np
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
import shiboken6
from unit_cell_gui import Atom, AtomComponent, Polyhedron, Scene, StyleOverrides, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from unit_cell_gui.gl_geometry import cylinder_mesh, indexed_cylinder_mesh, polygon_triangles
from test_unit_cell_gui import rotation, sample_scene


def cube_scene():
    base = sample_scene()
    vertices = np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
    faces = ((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
    cube = Polyhedron(np.zeros(3), vertices, faces, 'cube', 'Cube', base.atoms[0].components)
    shifted = Polyhedron([.5, .3, -.8], vertices + [.5,.3,-.8], faces, 'other', 'Other', base.atoms[1].components)
    return Scene(base.atoms, base.atom_centers, base.bonds, base.cell_vertices, base.cell_edges,
                 base.basis_vectors, (cube,shifted), base.base_radius)


class SceneInstanceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = UnitCellViewer()
        self.viewer.set_scene(cube_scene())

    def tearDown(self):
        self.viewer.close()
        shiboken6.delete(self.viewer)

    def test_shared_cylinder_and_frames_reproduce_half_bond_tessellation(self):
        local, indices = indexed_cylinder_mesh()
        self.assertEqual(len(indices), 72)
        self.assertEqual(indices.dtype, np.uint32)
        self.assertLess(len(local), len(indices))
        starts = np.array([[0,0,0],[.7,-.3,1.2],[-.5,.4,-1.1],[1,2,3]], float)
        deltas = np.array([[0,0,2],[.3,1.5,-.2],[.436,0,.9],[0,0,0]], float)
        component_a = AtomComponent('A','A','C',1,'#5588bb',.3)
        component_b = AtomComponent('B','B','O',1,'#ff8800',.4)
        atoms = tuple(Atom(str(i),str(i),(component_a if i%2==0 else component_b,)) for i in range(8))
        centers = np.stack((starts, starts+deltas), axis=1).reshape(-1,3)
        base = self.viewer.scene
        self.viewer.set_scene(Scene(atoms, centers, [[0,1],[2,3],[4,5],[6,7]],
                                    base.cell_vertices, base.cell_edges, base.basis_vectors, base_radius=4))
        records = self.viewer._build_colour_bond_instances()
        self.assertEqual(records.shape, (8,16))  # degenerate bonds retain clipped stable slots
        np.testing.assert_array_equal(records[6:,3:9], np.zeros((2,6)))
        radius = .08/3
        for i in range(3):
            a,b = centers[2*i:2*i+2]; mid=(a+b)*.5
            for half,(start,finish) in enumerate(((a,mid),(mid,b))):
                data=records[2*i+half]; radial=local[indices,:1]*data[3:6]+local[indices,1:2]*data[6:9]
                expanded=(1-local[indices,2:3])*data[:3]+local[indices,2:3]*data[9:12]+radial
                np.testing.assert_array_equal(data[9:12], np.asarray(finish,np.float32))
                expected,normals=cylinder_mesh(start,finish,radius)
                np.testing.assert_allclose(expanded,expected,atol=4e-7)
                np.testing.assert_allclose(radial/np.linalg.norm(data[3:6]),normals,atol=2e-7)
                np.testing.assert_allclose(data[12:16], [85/255,136/255,187/255,1] if half==0 else [1,136/255,0,1], atol=1e-6)

    def test_triangle_records_keep_original_corners_face_ranges_and_only_real_edges(self):
        faces_before = self.viewer.scene.polyhedra[0].faces
        geometry = self.viewer._build_static_geometry()
        records, ranges, edges = geometry[3], geometry[4], geometry[5]
        self.assertEqual(records.shape, (24,16))
        self.assertEqual(edges.shape, (24,12)) # 12 physical cube edges per polyhedron
        self.assertEqual(len(ranges),12)
        self.assertTrue(all(item.count==2 for item in ranges))
        expanded = np.stack((records[:,:3],records[:,4:7],records[:,8:11]),axis=1).reshape(-1,3)
        expected = np.concatenate([polygon_triangles(poly.vertices[list(face)],poly.center)[0]
                                   for poly in self.viewer.scene.polyhedra for face in poly.faces])
        np.testing.assert_array_equal(expanded, expected)
        self.assertEqual(self.viewer.scene.polyhedra[0].faces, faces_before)
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_opaque_sites={'cube'}))
        geometry = self.viewer._build_static_geometry()
        self.assertEqual(len(geometry[1]),12)
        self.assertEqual(len(geometry[3]),12)
        self.assertEqual(sum(item.count for item in geometry[7]),12)

    def test_bond_filters_and_engraving_endpoint_records(self):
        data = self.viewer._build_colour_bond_instances()
        self.assertEqual(len(data),2)
        lines = self.viewer._build_engraving_bond_instances()
        self.assertEqual(lines.shape,(1,12))
        np.testing.assert_array_equal(lines[0,:3], self.viewer.scene.atom_centers[0])
        np.testing.assert_array_equal(lines[0,4:7], self.viewer.scene.atom_centers[1])
        self.viewer.set_style_overrides(StyleOverrides(atom_component_visibility={
            c.key:False for atom in self.viewer.scene.atoms for c in atom.components}))
        self.assertEqual(len(self.viewer._build_colour_bond_instances()),0)
        self.assertEqual(len(self.viewer._build_engraving_bond_instances()),0)
        self.viewer.set_scene(None)
        self.assertEqual(self.viewer._build_static_geometry()[1].shape,(0,16))


class OpenGLSceneInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:
            raise unittest.SkipTest('A window system with OpenGL is required')

    def setUp(self):
        self.viewer=UnitCellViewer(); self.viewer.resize(640,480)
        self.viewer.set_scene(cube_scene()); self.viewer.set_display_options(show_polyhedra=True)
        self.viewer.set_orientation(rotation(0,19)@rotation(1,31))
        self.viewer.show(); self.app.processEvents()
        self.assertTrue(self.viewer.isValid()); self.assertIsNone(self.viewer.opengl_error)

    def tearDown(self):
        self.viewer.close(); shiboken6.delete(self.viewer); self.app.processEvents()

    def pixels(self):
        image=self.viewer.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.viewer.opengl_error); self.assertFalse(image.isNull())
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy()

    def test_gl_buffer_sizes_texture_ranges_and_draw_modes(self):
        from OpenGL.GL import (GL_ARRAY_BUFFER,GL_BUFFER_SIZE,GL_NO_ERROR,GL_ELEMENT_ARRAY_BUFFER_BINDING,
                               GL_VERTEX_ATTRIB_ARRAY_DIVISOR,glBindBuffer,glBindVertexArray,
                               glGetBufferParameteriv,glGetIntegerv,glGetVertexAttribiv,glGetError)
        self.pixels(); self.viewer.makeCurrent()
        try:
            glBindVertexArray(self.viewer._bond_vao)
            self.assertEqual(int(glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING)), self.viewer._bond_ebo)
            for location in range(6):
                self.assertEqual(np.asarray(glGetVertexAttribiv(location,GL_VERTEX_ATTRIB_ARRAY_DIVISOR)).ravel()[0],0 if location==0 else 1)
            for name,size in (('_bond_vbo',128),('_face_vbo',24*64),('_face_edge_vbo',24*48),('_engraving_bond_vbo',48)):
                glBindBuffer(GL_ARRAY_BUFFER,getattr(self.viewer,name))
                self.assertEqual(int(glGetBufferParameteriv(GL_ARRAY_BUFFER,GL_BUFFER_SIZE)),size)
            glBindVertexArray(0); glBindBuffer(GL_ARRAY_BUFFER,0)
            self.assertEqual(glGetError(),GL_NO_ERROR)
        finally: self.viewer.doneCurrent()
        with patch.object(renderer,'glDrawElementsInstanced',wraps=renderer.glDrawElementsInstanced) as cylinder, \
             patch.object(self.viewer,'_draw_primitive_instances',wraps=self.viewer._draw_primitive_instances) as primitive, \
             patch.object(renderer,'glDrawArraysInstanced',wraps=renderer.glDrawArraysInstanced) as lines:
            self.pixels()
            self.assertTrue(any(call.args[1]==72 and call.args[4]==2 for call in cylinder.call_args_list))
            tris=[c.args for c in primitive.call_args_list if c.args[0]==renderer.GL_TRIANGLES]
            self.assertEqual(sum(args[2] for args in tris),self.viewer._face_count)
            self.assertTrue(all(args[2]>0 for args in tris))
            edge_draws=[c for c in lines.call_args_list if c.args[:3]==(renderer.GL_LINES,0,2)]
            self.assertEqual(sum(c.args[3] for c in edge_draws),24)

    def test_camera_visibility_and_style_preserve_shared_meshes_and_restore_frame(self):
        before=self.pixels(); meshes=(self.viewer._bond_mesh_vbo,self.viewer._bond_ebo,self.viewer._primitive_mesh_vbo)
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as upload:
            self.viewer.set_orientation(rotation(0,44)@rotation(1,28)); self.viewer.zoom_by(1)
            self.pixels()
            self.viewer.set_style_overrides(StyleOverrides(polyhedron_site_visibility={'cube':False}))
            self.pixels(); upload.assert_not_called()
        self.viewer.set_style_overrides(StyleOverrides()); self.viewer.zoom_by(-1)
        self.viewer.set_orientation(rotation(0,19)@rotation(1,31))
        np.testing.assert_array_equal(self.pixels(),before)
        self.viewer.set_display_options(engraving=True)
        self.assertFalse(np.array_equal(self.pixels(),before))
        self.viewer.set_display_options(engraving=False)
        np.testing.assert_array_equal(self.pixels(),before)
        self.assertEqual(meshes,(self.viewer._bond_mesh_vbo,self.viewer._bond_ebo,self.viewer._primitive_mesh_vbo))

    def test_small_texture_capacity_keeps_transparency_order_and_releases_chunks(self):
        from OpenGL.GL import glIsTexture, glIsBuffer
        before = self.pixels()
        self.viewer._primitive_texel_capacity = 8  # two triangles or two segments per chunk
        self.viewer._gpu_dirty = self.viewer._bond_dirty = True
        np.testing.assert_array_equal(self.pixels(), before)
        extras = [(vbo, tex) for chunks in self.viewer._primitive_chunks.values()
                  for vbo, tex, _size in chunks[1:]]
        self.assertGreater(len(extras), 0)
        self.viewer.set_style_overrides(StyleOverrides(polyhedron_opaque_sites={'cube', 'other'}))
        self.pixels()  # an opaque range spans several texture buffers
        self.viewer.set_style_overrides(StyleOverrides())
        np.testing.assert_array_equal(self.pixels(), before)
        extras = [(vbo, tex) for chunks in self.viewer._primitive_chunks.values()
                  for vbo, tex, _size in chunks[1:]]
        self.viewer.cleanup(); self.viewer.makeCurrent()
        try:
            self.assertTrue(all(not glIsBuffer(vbo) and not glIsTexture(tex) for vbo,tex in extras))
        finally:
            self.viewer.doneCurrent()

    def test_cleanup_and_reinitialization_release_all_instance_resources(self):
        from OpenGL.GL import glIsBuffer,glIsTexture,glIsProgram
        before=self.pixels()
        buffers=[getattr(self.viewer,n) for n in ('_bond_vbo','_bond_mesh_vbo','_bond_ebo','_primitive_mesh_vbo','_face_vbo','_face_edge_vbo','_face_edge_order_vbo')]
        textures=[getattr(self.viewer,n) for n in ('_opaque_texture','_face_texture','_face_edge_texture','_engraving_bond_texture')]
        programs=[self.viewer._bond_program,self.viewer._triangle_program,self.viewer._segment_program]
        self.viewer.cleanup(); self.viewer.makeCurrent()
        try:
            self.assertTrue(all(not glIsBuffer(x) for x in buffers))
            self.assertTrue(all(not glIsTexture(x) for x in textures))
            self.assertTrue(all(not glIsProgram(x) for x in programs))
            self.viewer.initializeGL()
        finally: self.viewer.doneCurrent()
        np.testing.assert_array_equal(self.pixels(),before)
