"""Hybrid quality, signals, GL compositing, residency and resource ownership."""
import os
from dataclasses import replace
from unittest.mock import patch
import unittest
import numpy as np
import shiboken6
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from OpenGL import GL
from PySide6.QtCore import Qt, QPointF, QEvent, QCoreApplication
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from unit_cell_gui import DisplayOptions, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from unit_cell_gui import _weighted_oit as weighted
from render_test_helpers import adjacent_cubes, face, scene
from test_unit_cell_gui import sample_scene, rotation


def mouse(widget, kind, button, x=20, y=20, modifiers=Qt.KeyboardModifier.NoModifier):
    event=QMouseEvent(kind,QPointF(x,y),QPointF(x,y),button,button,modifiers)
    {QEvent.Type.MouseButtonPress:widget.mousePressEvent,
     QEvent.Type.MouseMove:widget.mouseMoveEvent,
     QEvent.Type.MouseButtonRelease:widget.mouseReleaseEvent}[kind](event)

class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.v=UnitCellViewer();self.v.set_scene(sample_scene())
        self.v.set_display_options(show_polyhedra=True)
    def tearDown(self):
        self.v.close();shiboken6.delete(self.v);self.app.processEvents()
    def enable(self):self.v.set_display_options(weighted_blended_oit_when_rotate=True)
    def move(self,axis=1,angle=20):self.v.set_orientation(rotation(axis,angle))
    def test_default_trailing_option_and_instance_isolation(self):
        self.assertFalse(DisplayOptions().weighted_blended_oit_when_rotate)
        old=[getattr(DisplayOptions(),k) for k in DisplayOptions.__dataclass_fields__ if k not in ('weighted_blended_oit_when_rotate','weighted_blended_oit')]
        self.assertFalse(DisplayOptions(*old).weighted_blended_oit_when_rotate)
        other=UnitCellViewer()
        try:
            self.enable();self.assertFalse(other.weighted_blended_oit_when_rotate)
            self.assertIsNone(self.v._weighted_oit)
            with self.assertRaises(TypeError):self.v.set_display_options(weighted_blended_oit_when_rotate='yes')
        finally:other.close();shiboken6.delete(other)
    def test_external_timer_same_orientation_emit_and_final_quality(self):
        signals=[];finished=[]
        self.v.orientation_changed.connect(lambda value:signals.append(value))
        self.v.interaction_finished.connect(lambda:finished.append(True))
        self.enable();self.move()
        self.assertTrue(self.v._should_use_weighted_oit());self.assertTrue(self.v._rotation_timer.isActive())
        QTest.qWait(60)
        before=self.v._rotation_timer.remainingTime()
        self.v.set_orientation(self.v.orientation,emit=True)
        self.assertLessEqual(self.v._rotation_timer.remainingTime(),before)
        self.assertEqual(len(signals),1)
        QTest.qWait(130)
        self.assertFalse(self.v._should_use_weighted_oit());self.assertTrue(self.v._awaiting_exact_frame)
        self.assertEqual(len(signals),1);self.assertEqual(finished,[])
        self.move(angle=40);self.assertFalse(self.v._awaiting_exact_frame)
        self.assertTrue(self.v._should_use_weighted_oit())
    def test_effectively_identical_float32_orientation_does_not_start_timer(self):
        self.enable();matrix=np.eye(3);matrix[0,0]+=1e-10
        self.v.set_orientation(matrix)
        self.assertFalse(self.v._rotation_timer.isActive())
        self.assertFalse(self.v._should_use_weighted_oit())
    def test_invalid_orientation_is_atomic_for_quality(self):
        self.enable()
        with self.assertRaises(ValueError):self.v.set_orientation(np.zeros((3,3)))
        self.assertFalse(self.v._rotation_timer.isActive());self.assertFalse(self.v._should_use_weighted_oit())
    def test_internal_drag_held_until_release_signals_and_no_timer(self):
        self.enable();signals=[];finished=[]
        self.v.orientation_changed.connect(lambda value:signals.append(value))
        self.v.interaction_finished.connect(lambda:finished.append(True))
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.LeftButton)
        self.assertFalse(self.v._should_use_weighted_oit())
        mouse(self.v,QEvent.Type.MouseMove,Qt.MouseButton.LeftButton,30,26)
        self.assertTrue(self.v._should_use_weighted_oit());self.assertFalse(self.v._rotation_timer.isActive())
        QTest.qWait(180);self.assertTrue(self.v._should_use_weighted_oit())
        mouse(self.v,QEvent.Type.MouseButtonRelease,Qt.MouseButton.LeftButton,30,26)
        self.assertFalse(self.v._should_use_weighted_oit());self.assertTrue(self.v._awaiting_exact_frame)
        self.assertEqual(len(signals),1);self.assertEqual(len(finished),1)
    def test_pan_zoom_disabled_rotation_and_engraving_do_not_start_quality(self):
        self.enable()
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.RightButton)
        mouse(self.v,QEvent.Type.MouseMove,Qt.MouseButton.RightButton,30,26)
        mouse(self.v,QEvent.Type.MouseButtonRelease,Qt.MouseButton.RightButton,30,26)
        self.v.zoom_by(1)
        self.assertFalse(self.v._should_use_weighted_oit());self.assertFalse(self.v._rotation_timer.isActive())
        self.v.set_display_options(rotation_enabled=False)
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.LeftButton)
        mouse(self.v,QEvent.Type.MouseMove,Qt.MouseButton.LeftButton,30,26)
        self.assertFalse(self.v._rotation_drag_active)
        self.v.set_display_options(engraving=True);self.move()
        self.assertFalse(self.v._should_use_weighted_oit());self.assertFalse(self.v._rotation_timer.isActive())
    def test_disable_clear_close_and_interruption_stop_motion(self):
        for operation in (lambda:self.v.set_display_options(weighted_blended_oit_when_rotate=False),
                          lambda:self.v.set_scene(None),lambda:self.v.close(),
                          lambda:QCoreApplication.sendEvent(self.v,QEvent(QEvent.Type.UngrabMouse)),
                          lambda:QCoreApplication.sendEvent(self.v,QEvent(QEvent.Type.WindowDeactivate))):
            self.v.set_scene(sample_scene());self.enable();self.move(angle=20)
            self.move(angle=30);self.assertTrue(self.v._rotation_timer.isActive())
            operation()
            self.assertFalse(self.v._rotation_timer.isActive());self.assertFalse(self.v._should_use_weighted_oit())
    def test_new_drag_interrupts_old_rotation_and_menu_tracks_external_option(self):
        self.enable()
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.LeftButton)
        mouse(self.v,QEvent.Type.MouseMove,Qt.MouseButton.LeftButton,30,26)
        self.assertTrue(self.v._should_use_weighted_oit())
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.RightButton)
        mouse(self.v,QEvent.Type.MouseButtonRelease,Qt.MouseButton.RightButton)
        self.assertFalse(self.v._should_use_weighted_oit())
        self.v._show_quality_menu(self.v.mapToGlobal(self.v.rect().center()))
        menu=self.v._quality_menu
        self.v.set_display_options(weighted_blended_oit_when_rotate=False)
        self.assertFalse(menu.actions()[0].isChecked())
        menu.close()

    def test_permanent_option_is_trailing_independent_and_boolean(self):
        options=DisplayOptions(weighted_blended_oit=True)
        self.assertTrue(options.weighted_blended_oit)
        self.assertFalse(options.weighted_blended_oit_when_rotate)
        old=[getattr(DisplayOptions(),name) for name in DisplayOptions.__dataclass_fields__ if name!='weighted_blended_oit']
        self.assertFalse(DisplayOptions(*old).weighted_blended_oit)
        self.v.set_display_options(options,show_polyhedra=True)
        self.assertTrue(self.v._should_use_weighted_oit())
        self.assertFalse(self.v._rotation_timer.isActive())
        for value in (1,'true',None):
            with self.assertRaises(TypeError):self.v.set_display_options(weighted_blended_oit=value)
        self.assertTrue(self.v.weighted_blended_oit)

    def test_permanent_wins_without_timer_or_pending_exact_and_keeps_rotation_preference(self):
        self.enable();self.move()
        self.v.set_display_options(weighted_blended_oit=True)
        self.assertFalse(self.v._rotation_timer.isActive())
        self.move(angle=40);self.assertFalse(self.v._rotation_timer.isActive())
        QTest.qWait(180);self.assertTrue(self.v._should_use_weighted_oit())
        self.assertFalse(self.v._awaiting_exact_frame)
        self.v._finish_mouse_rotation();self.assertTrue(self.v._should_use_weighted_oit())
        self.assertFalse(self.v._awaiting_exact_frame)
        self.v.set_display_options(weighted_blended_oit=False)
        self.assertTrue(self.v.weighted_blended_oit_when_rotate)
        self.assertFalse(self.v._should_use_weighted_oit())
        self.move(angle=45);self.assertTrue(self.v._rotation_timer.isActive())
        self.v._external_rotation_stopped();self.assertFalse(self.v._should_use_weighted_oit())

    def test_second_menu_action_round_trips_and_both_are_unavailable_on_failure(self):
        self.v._show_quality_menu(self.v.mapToGlobal(self.v.rect().center()));menu=self.v._quality_menu
        self.assertEqual(menu.actions()[1].text(),'Weighted Blended OIT always')
        menu.actions()[1].trigger();self.assertTrue(self.v.weighted_blended_oit)
        self.assertFalse(self.v.weighted_blended_oit_when_rotate)
        self.v.set_display_options(weighted_blended_oit=False,weighted_blended_oit_when_rotate=True)
        self.assertFalse(menu.actions()[1].isChecked());self.assertTrue(menu.actions()[0].isChecked())
        menu.close();self.v.weighted_oit_error='test failure'
        self.v._show_quality_menu(self.v.mapToGlobal(self.v.rect().center()));menu=self.v._quality_menu
        self.assertFalse(menu.actions()[0].isEnabled());self.assertFalse(menu.actions()[1].isEnabled())
        menu.close()

    def test_fallback_quality_and_menu_reflect_error(self):
        self.enable();self.move();self.v.weighted_oit_error='test framebuffer failure'
        self.assertFalse(self.v._should_use_weighted_oit())
        self.v._show_quality_menu(self.v.mapToGlobal(self.v.rect().center()))
        menu=self.v._quality_menu
        self.assertFalse(menu.actions()[0].isEnabled());self.assertTrue(menu.actions()[0].isChecked())
        self.assertIn('test framebuffer failure',menu.actions()[2].text());menu.close()
    def test_shift_right_menu_leaves_no_drag_and_applies_display_options(self):
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.RightButton,modifiers=Qt.KeyboardModifier.ShiftModifier)
        self.assertIsNone(self.v._last_mouse);self.assertIsNone(self.v._drag_button)
        menu=self.v._quality_menu
        menu.actions()[0].trigger()
        self.assertTrue(self.v.weighted_blended_oit_when_rotate)
        menu.close()
        mouse(self.v,QEvent.Type.MouseButtonRelease,Qt.MouseButton.RightButton)
        mouse(self.v,QEvent.Type.MouseButtonPress,Qt.MouseButton.RightButton)
        self.assertEqual(self.v._drag_button,Qt.MouseButton.RightButton)


class WeightedGLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])
        if cls.app.platformName() in {'offscreen','minimal'}:raise unittest.SkipTest('GL display required')
    def setUp(self):
        self.v=UnitCellViewer();self.v.resize(320,240)
        self.v.set_display_options(show_polyhedra=True,show_atoms=False,show_bonds=False,show_cell=False,show_basis=False)
        self.v.set_scene(scene((face('red','#ff0000',.4),face('green','#00ff00',-.4))))
        self.v.show();self.app.processEvents()
    def tearDown(self):self.v.close();shiboken6.delete(self.v);self.app.processEvents()
    def pixels(self):
        image=self.v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        self.assertIsNone(self.v.opengl_error);self.assertIsNone(self.v.weighted_oit_error)
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)
    def fast(self,angle=5):
        self.v.set_display_options(weighted_blended_oit_when_rotate=True)
        self.v.set_orientation(rotation(1,angle))
        image=self.pixels();self.assertTrue(self.v._last_frame_wboit)
        return image
    def gl_assert_clean(self):
        self.v.makeCurrent()
        try:self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:self.v.doneCurrent()
    def test_fast_has_no_sort_or_geometry_upload_and_final_frame_identical(self):
        self.fast()
        with patch.object(renderer,'back_to_front',side_effect=AssertionError('sort in fast frame')), \
             patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as upload, \
             patch.object(renderer,'glBufferSubData',wraps=renderer.glBufferSubData) as patch_upload:
            self.fast(17);upload.assert_not_called();patch_upload.assert_not_called()
        self.v._external_rotation_stopped();final=self.pixels();self.assertFalse(self.v._last_frame_wboit)
        self.v.set_display_options(weighted_blended_oit_when_rotate=False)
        np.testing.assert_array_equal(final,self.pixels());self.gl_assert_clean()
    def test_order_independence_background_and_opacities(self):
        polys=self.v.scene.polyhedra;self.v.set_polyhedron_opacities({0:.25,1:.6})
        first=self.fast();self.v._stop_camera_motion()
        self.v.set_scene(scene(polys[::-1]),reset_camera=False);self.v.set_polyhedron_opacities({0:.6,1:.25})
        second=self.fast(5.01)
        # Exactly same orientation without a setter event: start explicit series.
        self.v.set_orientation(rotation(1,5));second=self.pixels()
        np.testing.assert_allclose(second.astype(float),first.astype(float),atol=1)
        self.v.set_polyhedron_opacities({0:0,1:0});empty=self.pixels()
        colour=np.array(self.v.palette().base().color().getRgb())
        np.testing.assert_allclose(empty, np.broadcast_to(colour,empty.shape),atol=1)
        self.v.set_polyhedron_opacities({0:1,1:1});opaque_fast=self.pixels()
        self.v._stop_camera_motion();opaque_exact=self.pixels()
        # Filled interiors must be identical; edge blending rules are explicitly
        # separate from the former global edge/face list in the fast renderer.
        np.testing.assert_array_equal(opaque_fast[100:140,130:190],opaque_exact[100:140,130:190])
        self.gl_assert_clean()
    def test_many_low_alpha_layers_preserve_total_opacity(self):
        # R16F products near one round each (1-.0001) back to one. Use a
        # small projected face so this probes precision, not full-screen fill.
        poly=face('red','#ff0000',0)
        poly=replace(poly,vertices=poly.vertices*.12)
        self.v.set_scene(scene((poly,)*256))
        self.v.set_polyhedron_opacities({i:.0001 for i in range(256)})
        image=self.fast()
        centre=image[image.shape[0]//2,image.shape[1]//2,:3]
        expected=(np.array([255,0,0])*(1-(1-.0001)**256)
                  +np.array([255,255,255])*(1-.0001)**256)
        np.testing.assert_allclose(centre,expected,atol=1)

    def test_atom_msaa_and_contacts_are_not_lost_and_final_is_exact(self):
        self.v.set_scene(sample_scene());self.v.set_display_options(show_atoms=True,show_bonds=True)
        self.v.set_orientation(rotation(0,23)@rotation(1,31));exact=self.pixels()
        self.v.set_display_options(weighted_blended_oit_when_rotate=True)
        self.v.set_orientation(rotation(0,23)@rotation(1,32));fast=self.pixels()
        self.assertTrue(self.v._last_frame_wboit);self.assertGreater(self.v._intersection_count,0)
        self.v._stop_camera_motion();self.pixels()
        self.v.set_orientation(rotation(0,23)@rotation(1,31));self.v._stop_camera_motion()
        np.testing.assert_array_equal(self.pixels(),exact);self.gl_assert_clean()
    def test_resize_clear_close_reopen_resources_and_no_msaa(self):
        self.fast();oit=self.v._weighted_oit;handles=tuple(oit.textures);old_size=oit.size
        self.v.resize(501,401);self.fast(10)
        self.assertNotEqual(oit.size,old_size)
        self.assertEqual(oit.size[:2],self.pixels().shape[1::-1])
        self.v.set_scene(None);self.pixels();self.assertIsNone(oit.size)
        self.assertEqual(oit.attachment_bytes,0)
        self.v.set_scene(adjacent_cubes(2));self.fast(20)
        self.v.close();self.assertIsNone(self.v._weighted_oit)
        self.v.show();self.app.processEvents();self.fast(30);self.gl_assert_clean()
        other=UnitCellViewer();fmt=other.format();fmt.setSamples(0);other.setFormat(fmt)
        other.resize(201,173);other.set_scene(self.v.scene)
        other.set_display_options(show_polyhedra=True,show_cell=False,show_basis=False,weighted_blended_oit_when_rotate=True)
        try:
            other.show();self.app.processEvents();other.set_orientation(rotation(1,15));other.grabFramebuffer()
            self.assertIsNone(other.weighted_oit_error);self.assertTrue(other._last_frame_wboit)
            self.assertEqual(other._weighted_oit.size[2],1)
        finally:other.close();shiboken6.delete(other)
    def test_failed_initialization_falls_back_once_then_context_retry(self):
        self.v.set_orientation(rotation(1,13));expected=self.pixels()
        self.v.set_display_options(weighted_blended_oit_when_rotate=True)
        with patch.object(weighted.WeightedOIT,'initialize',side_effect=RuntimeError('injected failure')) as fail:
            self.v.set_orientation(rotation(1,14));self.v.grabFramebuffer()
            self.assertFalse(self.v._last_frame_wboit);self.assertIn('injected failure',self.v.weighted_oit_error)
            self.v.set_orientation(rotation(1,15));self.v.grabFramebuffer();self.assertEqual(fail.call_count,1)
        self.assertIsNone(self.v.opengl_error)
        self.v.close();self.v.show();self.app.processEvents();self.fast(16)
    def test_live_updates_and_opaque_edge_classification_in_fast_mode(self):
        current=sample_scene();self.v.set_scene(current)
        self.v.set_display_options(show_atoms=True,show_bonds=True)
        self.fast(17)
        poly=replace(current.polyhedra[0],vertices=current.polyhedra[0].vertices*1.08)
        point=current.atom_centers[1]+[.09,-.05,.02]
        self.v.update_scene(atom_positions={1:point},polyhedra={0:poly},polyhedron_opacities={0:.32})
        partial=self.pixels()
        rebuilt=replace(current,atom_centers=self.v.get_atom_positions(),polyhedra=self.v.get_polyhedra())
        self.v.set_scene(rebuilt,reset_camera=False);self.v.set_polyhedron_opacities({0:.32})
        self.v._external_rotation_active=True
        np.testing.assert_allclose(self.pixels().astype(float),partial.astype(float),atol=1)
        self.v.set_polyhedron_opacities({0:1})
        with patch.object(self.v,'_draw_weighted_ranges',wraps=self.v._draw_weighted_ranges) as draw:
            self.pixels()
        nonempty=[call.args for call in draw.call_args_list if call.args[0]]
        self.assertTrue(any(edge and program==self.v._segment_program for ranges,edge,program in nonempty))
        self.assertFalse(any(program in (self.v._weighted_oit.triangle_program,self.v._weighted_oit.segment_program)
                             for ranges,edge,program in nonempty))

    def test_native_mouse_release_emits_once_and_restores_exact_quality(self):
        self.v.set_display_options(weighted_blended_oit_when_rotate=True)
        finished=[];orientations=[]
        self.v.interaction_finished.connect(lambda:finished.append(True))
        self.v.orientation_changed.connect(lambda value:orientations.append(value))
        p=self.v.rect().center()
        QTest.mousePress(self.v,Qt.MouseButton.LeftButton,pos=p)
        QTest.mouseMove(self.v,p+self.v.rect().topLeft()+type(p)(11,7))
        self.pixels();self.assertTrue(self.v._last_frame_wboit)
        QTest.mouseRelease(self.v,Qt.MouseButton.LeftButton,pos=p+type(p)(11,7))
        self.pixels();self.assertFalse(self.v._last_frame_wboit)
        self.assertEqual(len(finished),1);self.assertEqual(len(orientations),1)

    def test_partial_attachment_failure_cleans_handles_and_limit_prevents_allocations(self):
        self.v.makeCurrent()
        try:
            oit=weighted.WeightedOIT()
            with weighted.preserved_state():
                with patch.object(weighted.g,'glTexImage2D',side_effect=RuntimeError('texture failure')), \
                     patch.object(weighted.g,'glDeleteTextures',wraps=GL.glDeleteTextures) as delete:
                    with self.assertRaises(RuntimeError):oit.ensure_size(32,32,1)
                    self.assertEqual(sum(call.args[0] for call in delete.call_args_list),1)
                self.assertFalse(oit.textures);self.assertFalse(oit.framebuffers);self.assertEqual(oit.depth,0)
                with patch.object(weighted,'MAX_ATTACHMENT_BYTES',16),patch.object(weighted.g,'glGenTextures') as create:
                    with self.assertRaises(RuntimeError):oit.ensure_size(32,32,1)
                    create.assert_not_called()
        finally:self.v.doneCurrent()
    def test_permanent_idle_and_release_skip_sort_and_toggle_back_to_exact(self):
        exact=self.pixels()
        with patch.object(renderer,'back_to_front',side_effect=AssertionError('sort in permanent mode')):
            self.v.set_display_options(weighted_blended_oit=True)
            permanent=self.pixels();self.assertTrue(self.v._last_frame_wboit)
            self.v._finish_mouse_rotation();self.pixels();self.assertTrue(self.v._last_frame_wboit)
            QTest.qWait(180);np.testing.assert_array_equal(self.pixels(),permanent)
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate, \
             patch.object(renderer,'glBufferSubData',wraps=renderer.glBufferSubData) as upload:
            self.v.set_display_options(weighted_blended_oit=False)
            np.testing.assert_array_equal(self.pixels(),exact)
            allocate.assert_not_called();upload.assert_not_called()
        self.assertFalse(self.v._last_frame_wboit)
        self.gl_assert_clean()

    def test_permanent_and_rotation_share_renderer_and_preserve_engraving(self):
        self.v.set_orientation(rotation(1,17));self.pixels()
        rotating=self.fast(18)
        self.v.set_display_options(weighted_blended_oit=True)
        self.assertFalse(self.v._rotation_timer.isActive())
        np.testing.assert_array_equal(self.pixels(),rotating)
        self.v.set_display_options(engraving=True);engraved=self.pixels()
        self.assertFalse(self.v._last_frame_wboit)
        self.v.set_display_options(weighted_blended_oit=False,weighted_blended_oit_when_rotate=False)
        np.testing.assert_array_equal(self.pixels(),engraved)
        self.v.set_display_options(engraving=False,weighted_blended_oit=True)
        self.pixels();self.assertTrue(self.v._last_frame_wboit)
        self.v.close();self.v.show();self.app.processEvents()
        self.pixels();self.assertTrue(self.v._last_frame_wboit)
        self.gl_assert_clean()

    def test_each_program_creation_failure_releases_all_previous_programs(self):
        self.v.makeCurrent()
        try:
            original=renderer._create_program
            for fail_at in range(1,5):
                created=[];calls=[];oit=weighted.WeightedOIT()
                def compile_program(*args):
                    calls.append(True)
                    if len(calls)==fail_at:raise RuntimeError('program fault')
                    program=original(*args);created.append(program);return program
                with patch.object(renderer,'_create_program',compile_program):
                    with self.assertRaises(RuntimeError):oit.initialize()
                self.assertTrue(all(not GL.glIsProgram(program) for program in created))
                self.assertEqual(oit.triangle_program,0);self.assertEqual(oit.segment_program,0)
                self.assertFalse(oit.composite_programs)
        finally:self.v.doneCurrent()

    def test_attachment_faults_release_every_registered_handle(self):
        self.v.makeCurrent()
        try:
            for function in ('glRenderbufferStorage','glTexImage2D','glFramebufferTexture2D',
                             'glFramebufferRenderbuffer','glCheckFramebufferStatus'):
                oit=weighted.WeightedOIT();textures=[];fbos=[];depth=[]
                original_t=GL.glGenTextures;original_f=GL.glGenFramebuffers;original_d=GL.glGenRenderbuffers
                def texture(*args):
                    handle=original_t(*args);textures.append(int(handle));return handle
                def framebuffer(*args):
                    handle=original_f(*args);fbos.append(int(handle));return handle
                def renderbuffer(*args):
                    handle=original_d(*args);depth.append(int(handle));return handle
                with weighted.preserved_state():
                    with patch.object(weighted.g,'glGenTextures',texture), \
                         patch.object(weighted.g,'glGenFramebuffers',framebuffer), \
                         patch.object(weighted.g,'glGenRenderbuffers',renderbuffer), \
                         patch.object(weighted.g,function,side_effect=RuntimeError('attachment fault')):
                        with self.assertRaises(RuntimeError):oit.ensure_size(32,32,1)
                self.assertTrue(all(not GL.glIsTexture(handle) for handle in textures))
                self.assertTrue(all(not GL.glIsFramebuffer(handle) for handle in fbos))
                self.assertTrue(all(not GL.glIsRenderbuffer(handle) for handle in depth))
        finally:self.v.doneCurrent()

    def test_failed_composition_restores_clean_exact_frame(self):
        self.fast(13);self.v._stop_camera_motion();expected=self.pixels()
        with patch.object(weighted.WeightedOIT,'composite',side_effect=RuntimeError('composition fault')):
            self.v._external_rotation_active=True
            image=self.v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        actual=np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)
        np.testing.assert_array_equal(actual,expected)
        self.assertIsNone(self.v.opengl_error);self.assertIn('composition fault',self.v.weighted_oit_error)
        self.assertFalse(self.v._last_frame_wboit)

    def test_passes_restore_framebuffer_viewport_and_texture_state(self):
        self.fast();self.v.makeCurrent()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,self.v.defaultFramebufferObject())
            GL.glViewport(7,9,123,117);GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glDepthMask(False);GL.glEnable(GL.GL_BLEND)
            before=(tuple(GL.glGetIntegerv(GL.GL_VIEWPORT)),int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)),
                    int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)))
            with weighted.preserved_state():
                self.v._weighted_oit.begin_scene((1,1,1,1))
                self.v._weighted_oit.begin_accumulation();self.v._weighted_oit.begin_revealage()
                self.v._weighted_oit.composite(self.v.defaultFramebufferObject())
            after=(tuple(GL.glGetIntegerv(GL.GL_VIEWPORT)),int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)),
                   int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)))
            self.assertEqual(before,after);self.assertFalse(GL.glGetBooleanv(GL.GL_DEPTH_WRITEMASK))
            self.assertTrue(GL.glIsEnabled(GL.GL_BLEND));self.assertEqual(GL.glGetError(),GL.GL_NO_ERROR)
        finally:self.v.doneCurrent()
