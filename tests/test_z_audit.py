"""Validation, lazy preparation, local geometry and resource-lifetime regressions."""
import os
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import shiboken6
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from OpenGL import GL
from unit_cell_gui import Atom, AtomComponent, DisplayOptions, Polyhedron, Scene, StyleOverrides, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from unit_cell_gui import _numeric as numeric
from unit_cell_gui._scene_updates import SceneView
from unit_cell_gui._spatial import BoundsIndex
from unit_cell_gui._transparency_sort import prepare_primitives, back_to_front, _project
from unit_cell_gui.hatching import occupancy_fractions
from test_unit_cell_gui import sample_scene, rotation
from test_visibility_cache import repeated_scene
from test_z_updates import separated_scene
from render_test_helpers import face


class SpatialTests(unittest.TestCase):
    def test_queries_and_refits_match_exhaustive_bounds(self):
        rng=np.random.default_rng(5252)
        low=rng.normal(size=(800,3))*20;high=low+rng.uniform(0,2,(800,3))
        low[7],high[7]=np.inf,-np.inf
        index=BoundsIndex(low,high)
        for step in range(40):
            ids=rng.choice(len(low),5,replace=False)
            low[ids]=rng.normal(size=(5,3))*20;high[ids]=low[ids]+rng.uniform(0,2,(5,3))
            index.update(ids,low[ids],high[ids])
            query_low=rng.normal(size=3)*15;query_high=query_low+5
            expected=np.flatnonzero(np.all(low<=query_high,axis=1)&np.all(high>=query_low,axis=1)&np.all(low<=high,axis=1))
            np.testing.assert_array_equal(index.query(query_low,query_high),expected)
        index.update([7],[[0,0,0]],[[1,1,1]])
        self.assertIn(7,index.query([0,0,0],[0,0,0]))
        empty=BoundsIndex(np.empty((0,3)),np.empty((0,3)))
        self.assertEqual(len(empty.query([0,0,0],[1,1,1])),0)

    def test_sparse_query_prunes_groups(self):
        low=np.column_stack((np.arange(4096)*4.,np.zeros((4096,2))))
        index=BoundsIndex(low,low+1)
        np.testing.assert_array_equal(index.query([0,0,0],[1,1,1]),[0])
        self.assertLess(index.last_visited,40)

    def test_scene_view_prepares_only_requested_geometry(self):
        base=separated_scene();view=SceneView(base)
        self.assertIs(view.atom_centers,base.atom_centers)
        self.assertIs(view.polyhedra,base.polyhedra)
        self.assertIsNone(view._bonds_by_atom);self.assertIsNone(view._bounds_index)
        view.opacity_overrides[0]=.2
        self.assertIsNone(view._atom_centers);self.assertIsNone(view._polyhedra)
        point=base.polyhedra[0].vertices[0]
        before=view.sections_near([(point,.1)])
        self.assertIn(0,before)
        index=view._bounds_index
        poly=replace(base.polyhedra[0],vertices=base.polyhedra[0].vertices+[100,0,0])
        view.writable_polyhedra()[0]=poly;view.update_bounds([0])
        self.assertIs(view._bounds_index,index)
        self.assertNotIn(0,view.sections_near([(point,.1)]))
        self.assertIn(0,view.sections_near([(point+[100,0,0],.1)]))
        self.assertIsNone(view._atom_centers);self.assertIsNone(view._bonds_by_atom)

    def test_prepared_sort_owns_inputs_and_preserves_general_transform_order(self):
        points=face('a','#ff0000',.3).vertices.copy()
        raw=[(False,points,('a',)),(False,points-[0,0,.5],('b',))]
        prepared=prepare_primitives(raw)
        for orientation in (np.eye(3),rotation(0,31)@rotation(1,19),np.array([[1,.2,0],[0,2,0],[0,0,-1]])):
            self.assertEqual(back_to_front(prepared,orientation),back_to_front(raw,orientation))
        before=back_to_front(prepared,np.eye(3));points[:]=100
        self.assertEqual(back_to_front(prepared,np.eye(3)),before)
        self.assertFalse(prepared.points.flags.writeable)

    def test_batched_projection_keeps_original_mean_depth_arithmetic(self):
        rng=np.random.default_rng(52521)
        primitives=[]
        for size in (2,3,4,5,8,9,16):
            for _ in range(12):
                primitives.append((size==2,rng.normal(size=(size,3)),('key',)))
        orientation=rotation(0,23)@rotation(1,37)
        projected,_=_project(prepare_primitives(primitives),orientation,.002,0.)
        for item,(_,points,_) in zip(projected,primitives):
            self.assertEqual(item.key[0],float((points@orientation.T)[:,2].mean()))

    def test_dense_diagnostics_include_retained_graph(self):
        stats=[]
        for count in (32,64):
            current={}
            primitives=[(False,face(str(i),'#4488cc',i*.1).vertices,(str(i),)) for i in range(count)]
            order,conflicts=back_to_front(prepare_primitives(primitives),np.eye(3),stats=current)
            self.assertEqual(conflicts,0);self.assertEqual(order,list(range(count)))
            self.assertEqual(current['candidates'],count*(count-1)//2)
            self.assertEqual(current['constraints'],count*(count-1)//2)
            stats.append(current)
        self.assertGreater(stats[1]['graph_bytes_estimate'],stats[0]['graph_bytes_estimate']*2)


class ShaderTests(unittest.TestCase):
    def test_failure_at_each_compile_stage_deletes_previous_shaders_once(self):
        for failure in range(3):
            results=[11,22,33][:failure]+[RuntimeError('compile failed')]
            with patch.object(renderer,'_compile_shader',side_effect=results), \
                 patch.object(renderer,'glDeleteShader') as delete, \
                 patch.object(renderer,'glCreateProgram') as create:
                with self.assertRaises(RuntimeError):renderer._create_program('v','f','g')
                self.assertEqual([call.args[0] for call in delete.call_args_list],[11,22,33][:failure])
                create.assert_not_called()

    def test_real_compile_helper_failure_deletes_every_created_handle_once(self):
        for failed_stage in range(3):
            with patch.object(renderer,'glCreateShader',side_effect=[11,22,33]) as create,                  patch.object(renderer,'glShaderSource'),patch.object(renderer,'glCompileShader'),                  patch.object(renderer,'glGetShaderiv',side_effect=[1]*failed_stage+[0]),                  patch.object(renderer,'glGetShaderInfoLog',return_value=b'compile failed'),                  patch.object(renderer,'glDeleteShader') as delete:
                with self.assertRaises(RuntimeError):renderer._create_program('v','f','g')
                self.assertEqual(create.call_count,failed_stage+1)
                self.assertEqual(sorted(call.args[0] for call in delete.call_args_list),[11,22,33][:failed_stage+1])

    def test_link_and_attach_failures_delete_program_and_all_shaders(self):
        for attach_failure in (False,True):
            with patch.object(renderer,'_compile_shader',side_effect=[11,22,33]), \
                 patch.object(renderer,'glCreateProgram',return_value=77), \
                 patch.object(renderer,'glAttachShader',side_effect=RuntimeError('attach') if attach_failure else None), \
                 patch.object(renderer,'glLinkProgram'),patch.object(renderer,'glGetProgramiv',return_value=0), \
                 patch.object(renderer,'glGetProgramInfoLog',return_value=b'link failed'), \
                 patch.object(renderer,'glDeleteShader') as shaders,patch.object(renderer,'glDeleteProgram') as programs:
                with self.assertRaises(RuntimeError):renderer._create_program('v','f','g')
                self.assertEqual([call.args[0] for call in shaders.call_args_list],[11,22,33])
                programs.assert_called_once_with(77)

    def test_shader_setup_exception_deletes_its_handle(self):
        with patch.object(renderer,'glCreateShader',return_value=42), \
             patch.object(renderer,'glShaderSource',side_effect=RuntimeError('source')), \
             patch.object(renderer,'glDeleteShader') as delete:
            with self.assertRaises(RuntimeError):renderer._compile_shader(GL.GL_VERTEX_SHADER,'v')
            delete.assert_called_once_with(42)


class AuditApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):self.v=UnitCellViewer();self.v.set_scene(separated_scene())
    def tearDown(self):self.v.close();shiboken6.delete(self.v)

    def test_invalid_orientations_are_atomic_and_general_matrices_remain_supported(self):
        before=self.v.orientation.copy();received=[];self.v.orientation_changed.connect(received.append)
        for matrix in (np.zeros((3,3)),np.diag([1,1,1e-12]),np.eye(3)*1e-50,np.eye(3)*1e100):
            with patch.object(self.v,'update') as repaint:
                with self.assertRaises(ValueError):self.v.set_orientation(matrix,emit=True)
                repaint.assert_not_called()
            np.testing.assert_array_equal(self.v.orientation,before)
        self.assertEqual(received,[])
        valid=np.array([[1,.2,0],[0,2,0],[0,0,-1.]])
        self.v.set_orientation(valid)
        np.testing.assert_array_equal(self.v.orientation,valid)
        self.v.set_display_options(engraving=True)
        self.assertTrue(np.all(np.isfinite(self.v._build_polyhedron_hatch_geometry(0,5))))

    def test_models_reject_lossy_indices_before_conversion(self):
        base=sample_scene();poly=base.polyhedra[0]
        for bad in (.9,'0',False,np.bool_(True)):
            with self.assertRaises(TypeError):replace(poly,faces=((bad,1,2),))
            with self.assertRaises(TypeError):replace(base,bonds=[[bad,1]])
            with self.assertRaises(TypeError):replace(base,cell_edges=[[bad,1]])
        for bad in (-1,2**100):
            with self.assertRaises(ValueError):replace(base,bonds=[[bad,1]])
        result=replace(base,bonds=np.array([[np.int64(0),np.int32(1)]]))
        np.testing.assert_array_equal(result.bonds,[[0,1]])
        self.assertEqual(result.bonds.dtype,np.dtype(int))

    def test_initial_and_computed_float32_overflows_are_rejected_atomically(self):
        base=sample_scene()
        with self.assertRaises(ValueError):replace(base,atom_centers=np.full_like(base.atom_centers,1e100))
        with self.assertRaises(ValueError):replace(base.atoms[0].components[0],radius=1e100)
        with self.assertRaises(ValueError):DisplayOptions(atom_scale=1e100)
        component=AtomComponent('large','large','C',1,'#5588bb',1e38)
        large=Scene((Atom('a','a',(component,)),),[[0,0,0]],[],[],[],np.eye(3),base_radius=1e38)
        self.v.set_scene(large)
        previous=self.v.scene
        with self.assertRaises(ValueError):self.v.set_display_options(atom_scale=4)
        self.assertEqual(self.v.atom_scale,1)
        self.v.set_display_options(atom_scale=4e-1)
        self.assertTrue(np.all(np.isfinite(self.v._build_atom_instances())))
        before=self.v.orientation.copy()
        with self.assertRaises(ValueError):self.v.set_orientation(np.diag([100.,1,1]))
        np.testing.assert_array_equal(self.v.orientation,before)
        with self.assertRaises(ValueError):self.v.set_atom_positions({0:[3.4e38,0,0]})
        np.testing.assert_array_equal(self.v.get_atom_positions(),[[0,0,0]])
        with self.assertRaises(ValueError):self.v.set_scene(replace(base,base_radius=1e-100))
        self.assertIs(self.v.scene,previous)

    def test_representable_values_near_float32_limit_keep_finite_records(self):
        limit=float(np.finfo(np.float32).max)
        scene=Scene((),[],[],[[0,0,0],[limit*.9,0,0]],[[0,1]],np.eye(3),base_radius=limit)
        self.v.set_scene(scene)
        geometry=self.v._build_static_geometry()
        self.assertTrue(np.all(np.isfinite(geometry[2])))
        self.assertTrue(np.all(np.isfinite(self.v._projection_matrix())))

    def test_large_finite_occupancies_remain_finite(self):
        first=AtomComponent('a','a','C',1e308,'#ff0000',.2)
        second=replace(first,key='b',colour='#0000ff',radius=.4)
        atom=Atom('a','a',(first,second))
        current=Scene((atom,),[[0,0,0]],[],[],[],np.eye(3))
        self.v.set_scene(current)
        np.testing.assert_array_equal(occupancy_fractions(atom.components),[.5,.5])
        self.assertAlmostEqual(self.v._atom_radius(atom),.3)
        for engraving in (False,True):
            self.v.set_display_options(engraving=engraving)
            self.assertTrue(np.all(np.isfinite(self.v._build_atom_instances())))
            self.assertTrue(np.all(np.isfinite(self.v._build_atom_outline_instances())))

    def test_reusing_scene_uses_cached_numeric_summary(self):
        current=self.v.scene
        with patch.object(numeric,'atom_radius',side_effect=AssertionError('radii scanned')):
            self.v.set_scene(current)
            self.v.set_display_options(atom_scale=1.2)
        self.assertFalse(current._numeric_bounds[0].flags.writeable)
        self.assertFalse(current._numeric_bounds[1].flags.writeable)

    def test_opacity_only_update_does_not_prepare_geometry(self):
        self.v.set_polyhedron_opacities({0:.3})
        view=self.v._scene_updates
        self.assertIsNone(view._atom_centers);self.assertIsNone(view._polyhedra)
        self.assertIsNone(view._bonds_by_atom);self.assertIsNone(view._bounds_index)
        self.v.set_polyhedron_opacities({0:.4})
        self.assertIs(self.v._scene_updates,view)
        self.assertAlmostEqual(self.v.get_polyhedron_opacities()[0],.4)

    def test_contact_index_matches_exhaustive_scan_and_refits_only_changed_atoms(self):
        self.v.set_scene(repeated_scene(20));self.v.set_display_options(show_polyhedra=True)
        for step in range(3):
            if step:
                self.v.update_scene(atom_positions={1:[1+step*.07,.03,0]},atom_visibility={3:step==2})
            indexed,ranges=self.v._build_atom_face_intersection_geometry()
            cache=self.v._contact_atoms
            with patch.object(cache,'near_face',side_effect=lambda low,high:np.flatnonzero(cache.visible)):
                exhaustive,expected=self.v._build_atom_face_intersection_geometry()
            np.testing.assert_allclose(indexed,exhaustive,atol=1e-7)
            self.assertEqual([(r.poly_index,r.count) for r in ranges],[(r.poly_index,r.count) for r in expected])
        centres=cache.centres;index=cache.index
        with patch.object(self.v,'_atom_radius',wraps=self.v._atom_radius) as radius:
            self.v.set_atom_positions({1:[1.1,.1,0]})
            self.v._build_atom_face_intersection_geometry(poly_indices=(0,))
            self.assertLessEqual(radius.call_count,3)
        self.assertIs(self.v._contact_atoms,cache);self.assertIs(cache.centres,centres);self.assertIs(cache.index,index)

    def test_legend_uses_visible_components_and_reuses_entries(self):
        a=AtomComponent('hidden','hidden','C',.5,'#ff0000',.2)
        b=replace(a,key='visible',colour='#0000ff')
        self.v.set_scene(Scene((Atom('a','a',(a,b)),),[[0,0,0]],[],[],[],np.eye(3)))
        self.v.set_style_overrides(StyleOverrides(atom_component_visibility={'hidden':False}))
        entries=self.v._legend_entries()
        self.assertEqual([(element,colour.name()) for element,colour in entries],[('C','#0000ff')])
        with patch.object(self.v,'_component_visible',side_effect=AssertionError('legend rescanned')):
            self.assertIs(self.v._legend_entries(),entries)
        self.v.set_atom_visibility({0:False});self.assertEqual(self.v._legend_entries(),())
        self.v.reset_scene_updates();self.assertEqual(self.v._legend_entries()[0][1].name(),'#0000ff')
        self.v.set_style_overrides(StyleOverrides(atom_component_visibility={'hidden':False},atom_component_colours={'visible':'#00ff00'}))
        self.assertEqual(self.v._legend_entries()[0][1].name(),'#00ff00')

    def test_cleanup_make_current_failure_can_be_retried(self):
        before=self.v._program
        with patch.object(self.v,'context',return_value=SimpleNamespace(isValid=lambda:True)), \
             patch.object(self.v,'makeCurrent',side_effect=RuntimeError('context')):
            with self.assertRaises(RuntimeError):self.v.cleanup()
        self.assertFalse(self.v._cleaning_up);self.assertEqual(self.v._program,before)


class AuditGLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('OpenGL display required')
    def setUp(self):
        self.v=UnitCellViewer();self.v.resize(640,480);self.scene=separated_scene()
        self.v.set_scene(self.scene);self.v.set_display_options(show_polyhedra=True)
        self.v.show();self.app.processEvents();self.pixels()
    def tearDown(self):self.v.close();shiboken6.delete(self.v);self.app.processEvents()
    def pixels(self,viewer=None):
        current=viewer or self.v
        image=current.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(current.opengl_error)
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy()
    def reference(self,style):
        other=UnitCellViewer();other.resize(640,480);other.set_scene(self.scene)
        other.set_display_options(show_polyhedra=True,engraving=self.v.engraving)
        other.set_style_overrides(style);other.show();self.app.processEvents()
        try:np.testing.assert_array_equal(self.pixels(),self.pixels(other))
        finally:other.close();shiboken6.delete(other);self.app.processEvents()

    def test_clear_releases_main_storage_but_keeps_shared_meshes_and_programs(self):
        before=self.pixels();program=self.v._program;mesh=self.v._atom_mesh_vbo
        handles=[getattr(self.v,name) for name in ('_atom_vbo','_bond_vbo','_opaque_vbo','_face_vbo','_face_edge_vbo','_cell_vbo','_atom_outline_vbo','_intersection_vbo','_engraving_bond_vbo','_poly_opacity_vbo')]
        self.v.set_scene(None)
        self.assertEqual(self.v._atom_instance_data.nbytes,0)
        self.pixels();self.v.makeCurrent()
        try:
            for handle in handles:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER,handle)
                self.assertEqual(int(GL.glGetBufferParameteriv(GL.GL_ARRAY_BUFFER,GL.GL_BUFFER_SIZE)),0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER,mesh)
            self.assertGreater(int(GL.glGetBufferParameteriv(GL.GL_ARRAY_BUFFER,GL.GL_BUFFER_SIZE)),0)
            self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:self.v.doneCurrent()
        self.assertEqual(self.v._program,program)
        self.assertIsNone(self.v._transparent_prepared);self.assertIsNone(self.v._contact_atoms)
        self.v.set_scene(self.scene)
        np.testing.assert_array_equal(self.pixels(),before)

    def test_host_style_refresh_is_local_and_matches_full_rebuild(self):
        for engraving in (False,True):
            self.v.set_display_options(engraving=engraving);self.pixels()
            style=StyleOverrides(polyhedron_site_colours={'cube':'#ff0088'},polyhedron_opaque_sites={'cube'})
            with patch.object(self.v,'_rebuild_gpu_cache',wraps=self.v._rebuild_gpu_cache) as full, \
                 patch.object(self.v,'_build_static_geometry',wraps=self.v._build_static_geometry) as geometry, \
                 patch.object(self.v,'_build_atom_face_intersection_geometry',wraps=self.v._build_atom_face_intersection_geometry) as contacts:
                self.v.set_style_overrides(style);self.pixels()
                full.assert_not_called()
                self.assertTrue(geometry.call_args_list)
                self.assertTrue(all(call.kwargs['poly_indices']==(0,) for call in geometry.call_args_list))
                self.assertTrue(all(call.kwargs['poly_indices']==(0,) for call in contacts.call_args_list))
            self.reference(style)
            self.v.set_style_overrides(StyleOverrides());self.pixels();self.reference(StyleOverrides())

    def test_rotation_and_opacity_reuse_world_preparation(self):
        with patch.object(renderer,'prepare_primitives',wraps=renderer.prepare_primitives) as prepare:
            prepared=self.v._transparent_prepared
            self.v.set_orientation(rotation(0,19)@rotation(1,31));self.pixels()
            self.v.zoom_by(1);self.pixels()
            prepare.assert_not_called();self.assertIs(self.v._transparent_prepared,prepared)
            self.v.set_polyhedron_opacities({0:.35});self.pixels()
            prepare.assert_not_called()
            self.v.set_atom_positions({1:[.6,.1,0]});self.pixels()
            prepare.assert_not_called()
            points=self.v._transparent_prepared.points
            self.v.set_style_overrides(StyleOverrides(polyhedron_site_colours={'cube':'#ff0088'}));self.pixels()
            prepare.assert_not_called()
            self.assertIs(self.v._transparent_prepared.points,points)
