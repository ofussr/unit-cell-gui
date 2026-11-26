"""Compare native versus legacy draw submission on the same live GL context.
No readback, camera/geometry setup and warm-up outside measured intervals.
Software GL CPU submit time includes synchronous driver work; completed time
includes glFinish. Neither is a measurement of hardware GPU time or FPS.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from render_test_helpers import adjacent_cubes
from test_unit_cell_gui import rotation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=12);args=p.parse_args()
    app=QApplication([]); rows=[];meta={}
    for count in (2,12,48,128):
        v=UnitCellViewer();v.resize(640,480);v.set_scene(adjacent_cubes(count));v.set_display_options(show_polyhedra=True,show_atoms=False,show_bonds=False,show_cell=False,show_basis=False)
        v.set_orientation(rotation(0,23)@rotation(1,37));v.show();app.processEvents();v.makeCurrent()
        assert v._native_gl is not None, v.native_render_error
        native=v._native_gl
        meta={'renderer':GL.glGetString(GL.GL_RENDERER).decode(),'version':GL.glGetString(GL.GL_VERSION).decode(),'samples':int(GL.glGetIntegerv(GL.GL_SAMPLES))}
        try:
            for weighted in (False,True):
                v.set_display_options(weighted_blended_oit=weighted)
                data={b:{'submit_ms':[],'completed_ms':[],'native_draws':[]} for b in ('python','rust')}
                def sample(b):
                    v._native_gl=native if b=='rust' else None;v._native_draw_calls=0
                    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject())
                    v._restore_render_state();GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT);GL.glFinish()
                    start=time.perf_counter()
                    if weighted: assert v._draw_weighted_frame(v.palette().base().color()),v.weighted_oit_error
                    else:v._draw_exact_scene()
                    submit=time.perf_counter();GL.glFinish();end=time.perf_counter()
                    return (submit-start)*1000,(end-start)*1000,v._native_draw_calls
                for b in ('python','rust'):sample(b)
                for i in range(args.samples):
                    for b in (('python','rust') if i%2==0 else ('rust','python')):
                        a,c,d=sample(b);data[b]['submit_ms'].append(a);data[b]['completed_ms'].append(c);data[b]['native_draws'].append(d)
                for b in data:
                    for k in ('submit_ms','completed_ms'):
                        data[b][k+'_p50']=float(np.median(data[b][k]));data[b][k+'_p95']=float(np.percentile(data[b][k],95))
                row={'polyhedra':count,'weighted':weighted,'measurements':data};rows.append(row)
                print(count,weighted,[(b,round(data[b]['submit_ms_p50'],3),round(data[b]['completed_ms_p50'],3)) for b in data],flush=True)
        finally:
            v._native_gl=native;v.doneCurrent();v.close();shiboken6.delete(v);app.processEvents()
    args.output.write_text(json.dumps({'metric':'fixed orientation draw exact or WBOIT scene, no readback or overlay; warmed caches','environment':meta,'cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
