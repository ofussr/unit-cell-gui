"""Native rotation benchmark: no framebuffer readback in the measured loop.

Exact and hybrid runs use the same scenes and camera sequence. GL query results
are collected after the run, never synchronously in a measured frame. CPU
measurements include instrumentation and submission, not completed GPU work.
"""
import argparse
import ctypes
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import shiboken6
from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_3_3 import glGetQueryObjectui64v
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
import unit_cell_gui
from unit_cell_gui import UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from unit_cell_gui import _weighted_oit as weighted
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from render_test_helpers import adjacent_cubes
from test_unit_cell_gui import rotation, sample_scene


def summary(samples):
    return {'samples':samples,'p50':float(np.percentile(samples,50)) if samples else None,
            'p95':float(np.percentile(samples,95)) if samples else None}


class BenchViewer(UnitCellViewer):
    def __init__(self):
        super().__init__();self.pending=None;self.records=[];self.queries=[]
        self.timer_bits=0;self.draws=0;self.sorts=0;self.uploads=0;self.setter_ms=0
    def paintGL(self):
        if self.pending is None:
            return super().paintGL()
        step=self.pending;self.pending=None
        self.draws=self.sorts=self.uploads=0
        query=0
        if self.timer_bits:
            query=int(np.asarray(GL.glGenQueries(1)).ravel()[0])
            GL.glBeginQuery(GL.GL_TIME_ELAPSED,query)
        start=time.perf_counter()
        try:super().paintGL()
        finally:
            elapsed=(time.perf_counter()-start)*1000
            if query:GL.glEndQuery(GL.GL_TIME_ELAPSED)
        self.records.append({'step':step,'cpu_prepare_submit_ms':elapsed+self.setter_ms,
                             'draw_calls':self.draws,'sort_calls':self.sorts,'geometry_upload_bytes':self.uploads,
                             'weighted':self._last_frame_wboit,'paint_end':time.perf_counter()})
        if query:self.queries.append((step,query))


def run(app,current,hybrid,args):
    viewer=BenchViewer();viewer.resize(args.width,args.height)
    viewer.set_scene(current)
    viewer.set_display_options(show_polyhedra=True,show_cell=False,show_basis=False,
                               weighted_blended_oit_when_rotate=hybrid)
    viewer.show();app.processEvents()
    viewer.makeCurrent()
    gpu={'renderer':GL.glGetString(GL.GL_RENDERER).decode(),'version':GL.glGetString(GL.GL_VERSION).decode()}
    try:viewer.timer_bits=int(GL.glGetQueryiv(GL.GL_TIME_ELAPSED,GL.GL_QUERY_COUNTER_BITS))
    except Exception:viewer.timer_bits=0
    timer_bits=viewer.timer_bits
    viewer.doneCurrent()
    loop=QEventLoop();swaps=[];steps=iter(range(args.frames+3));done=False
    def next_frame():
        nonlocal done
        if done:return
        try:step=next(steps)
        except StopIteration:
            done=True;loop.quit();return
        # Model a held mouse rotation, so the 150 ms external-motion timeout
        # cannot end a frame merely because a slow GPU has stalled the GUI.
        viewer._rotation_drag_active=True;viewer._orientation_source_mouse=True
        start=time.perf_counter();viewer.set_orientation(rotation(0,19)@rotation(1,23+step*.8))
        viewer.setter_ms=(time.perf_counter()-start)*1000
        viewer._orientation_source_mouse=False;viewer.pending=step;viewer.update()
    last_swap=[-1]
    def swapped():
        if not viewer.records:return
        step=viewer.records[-1]['step']
        if step==last_swap[0]:return
        last_swap[0]=step;swaps.append((step,time.perf_counter()))
        QTimer.singleShot(0,next_frame)
    viewer.frameSwapped.connect(swapped)
    # Track submission calls separately from GPU timings and swap intervals.
    with ExitStack() as stack:
        for module,name in ((renderer,'glDrawArrays'),(renderer,'glDrawArraysInstanced'),
                            (renderer,'glDrawElementsInstanced'),(weighted.g,'glDrawArrays')):
            original=getattr(module,name)
            def draw(*a,_original=original,**kw):
                viewer.draws+=1;return _original(*a,**kw)
            stack.enter_context(patch.object(module,name,draw))
        original_sort=renderer.back_to_front
        def sort(*a,**kw):viewer.sorts+=1;return original_sort(*a,**kw)
        stack.enter_context(patch.object(renderer,'back_to_front',sort))
        for name,size_arg in (('glBufferData',1),('glBufferSubData',2)):
            original=getattr(renderer,name)
            def upload(*a,_original=original,_size=size_arg,**kw):
                viewer.uploads+=int(a[_size]);return _original(*a,**kw)
            stack.enter_context(patch.object(renderer,name,upload))
        QTimer.singleShot(0,next_frame)
        timeout=QTimer();timeout.setSingleShot(True);timeout.timeout.connect(loop.quit)
        timeout.start(120000);loop.exec();timeout.stop()
        if not done:raise RuntimeError('benchmark timed out')
        # QWidget.repaint() can enqueue a QOpenGLWidget paint instead of
        # finishing it. Measure through frameSwapped, never just that call.
        final_loop=QEventLoop();final_elapsed=[]
        def final_presented():
            if (viewer.records and viewer.records[-1]['step']==args.frames+3
                    and not viewer._last_frame_wboit):
                final_elapsed.append((time.perf_counter()-final_start)*1000);final_loop.quit()
        viewer.frameSwapped.connect(final_presented)
        viewer.pending=args.frames+3;viewer.setter_ms=0
        final_start=time.perf_counter();viewer._finish_mouse_rotation()
        final_timeout=QTimer();final_timeout.setSingleShot(True);final_timeout.timeout.connect(final_loop.quit)
        final_timeout.start(10000);final_loop.exec();final_timeout.stop()
        viewer.frameSwapped.disconnect(final_presented)
        if not final_elapsed:raise RuntimeError('final exact frame was not presented')
        final_ms=final_elapsed[0]
    viewer.frameSwapped.disconnect(swapped)
    records=[r for r in viewer.records if 3<=r['step']<args.frames+3]
    assert len(records)==args.frames,(len(records),args.frames)
    assert viewer.opengl_error is None,viewer.opengl_error
    assert viewer.weighted_oit_error is None,viewer.weighted_oit_error
    assert all(r['weighted']==hybrid for r in records)
    if hybrid:assert all(r['sort_calls']==0 and r['geometry_upload_bytes']==0 for r in records)
    intervals=[(b[1]-a[1])*1000 for a,b in zip(swaps,swaps[1:]) if 3<=b[0]<args.frames+3]
    viewer.makeCurrent()
    gpu_times=[]
    try:
        GL.glFinish()  # outside the measured rotation loop
        for step,query in viewer.queries:
            if 3<=step<args.frames+3:
                # uint64 retrieval needs an explicit result array on PyOpenGL.
                value=ctypes.c_uint64()
                glGetQueryObjectui64v(query,GL.GL_QUERY_RESULT,ctypes.byref(value))
                gpu_times.append(float(value.value)/1e6)
            GL.glDeleteQueries(1,[query])
        assert GL.glGetError()==GL.GL_NO_ERROR
    finally:viewer.doneCurrent()
    idle_loop=QEventLoop();idle=[];viewer.timer_bits=0
    # A frameSwapped observation includes timer wait, exact paint and Qt present.
    def final_swapped():
        if not viewer._should_use_weighted_oit() and not viewer._last_frame_wboit:
            idle.append((time.perf_counter()-idle_start)*1000);idle_loop.quit()
    viewer.frameSwapped.connect(final_swapped)
    viewer.set_orientation(rotation(1,17));idle_start=time.perf_counter()
    idle_timeout=QTimer();idle_timeout.setSingleShot(True);idle_timeout.timeout.connect(idle_loop.quit)
    idle_timeout.start(10000);idle_loop.exec();idle_timeout.stop()
    viewer.frameSwapped.disconnect(final_swapped)
    result={'hybrid':hybrid,'cpu_prepare_submit_ms':summary([r['cpu_prepare_submit_ms'] for r in records]),
            'frame_interval_ms':summary(intervals),'gpu_elapsed_ms':summary(gpu_times),
            'timer_query_bits':timer_bits,
            'draw_calls':summary([r['draw_calls'] for r in records]),
            'sort_calls':sum(r['sort_calls'] for r in records),
            'geometry_upload_bytes':sum(r['geometry_upload_bytes'] for r in records),
            'release_to_exact_present_ms':final_ms,
            'final_exact_sort_calls':viewer.records[-1]['sort_calls'],'external_idle_to_exact_present_ms':idle,
            'extra_attachment_bytes':viewer._weighted_oit.attachment_bytes if viewer._weighted_oit else 0,
            'attachment_size':viewer._weighted_oit.size if viewer._weighted_oit else None,
            'records':records,'gpu':gpu}
    viewer.close();shiboken6.delete(viewer);app.processEvents()
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=60)
    parser.add_argument('--width',type=int,default=640);parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--counts',type=int,nargs='+',default=[2,12,48])
    args=parser.parse_args()
    if args.frames<4:parser.error('at least four frames required')
    app=QApplication([])
    if app.platformName() in {'offscreen','minimal'}:raise RuntimeError('native OpenGL display required')
    result={'version':unit_cell_gui.__version__,'readback_in_measured_loop':False,'cases':{}}
    for count in args.counts:
        current=adjacent_cubes(count)
        result['cases'][f'polyhedra{count}']={}
        for hybrid in (False,True):
            data=run(app,current,hybrid,args)
            result['cases'][f'polyhedra{count}']['hybrid' if hybrid else 'exact']=data
            print(count,hybrid,data['cpu_prepare_submit_ms']['p50'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
