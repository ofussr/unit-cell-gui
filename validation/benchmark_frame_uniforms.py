"""a6 UBO + native pass/state vs a5 uniform calls; both use Rust draws."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from render_test_helpers import adjacent_cubes
from test_unit_cell_gui import rotation,sample_scene

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=12);args=p.parse_args()
    app=QApplication([]); rows=[];meta={}
    for count in (1,12,48,128):
        viewers={}
        try:
            for backend in ('python','rust'):
                os.environ['UNIT_CELL_GUI_FRAME_BACKEND']=backend
                v=UnitCellViewer();viewers[backend]=v;v.resize(640,480);v.set_scene(sample_scene() if count==1 else adjacent_cubes(count));v.set_display_options(show_polyhedra=True,show_atoms=count==1,show_bonds=count==1,show_cell=False,show_basis=False)
                v.set_orientation(rotation(0,23)@rotation(1,37));v.show();app.processEvents();v.makeCurrent()
                assert v._native_gl is not None, v.native_render_error
                assert (v._frame_parameters is not None)==(backend=='rust'),v.frame_uniform_error
                meta={'renderer':GL.glGetString(GL.GL_RENDERER).decode(),'version':GL.glGetString(GL.GL_VERSION).decode(),'samples':int(GL.glGetIntegerv(GL.GL_SAMPLES))}
                v.doneCurrent()
            for weighted in (False,True):
                for v in viewers.values():v.set_display_options(weighted_blended_oit=weighted)
                data={b:{'submit_ms':[],'completed_ms':[]} for b in viewers}
                def sample(b):
                    v=viewers[b];v.makeCurrent()
                    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject());v._restore_render_state();GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT);GL.glFinish()
                    start=time.perf_counter()
                    if weighted: assert v._draw_weighted_frame(v.palette().base().color()),v.weighted_oit_error
                    else:v._draw_exact_scene()
                    submit=time.perf_counter();GL.glFinish();end=time.perf_counter();v.doneCurrent()
                    return (submit-start)*1000,(end-start)*1000
                for b in viewers:sample(b)
                for i in range(args.samples):
                    for b in (('python','rust') if i%2==0 else ('rust','python')):
                        a,c=sample(b);data[b]['submit_ms'].append(a);data[b]['completed_ms'].append(c)
                for b in data:
                    for k in ('submit_ms','completed_ms'):
                        data[b][k+'_p50']=float(np.median(data[b][k]));data[b][k+'_p95']=float(np.percentile(data[b][k],95))
                row={'polyhedra':count,'weighted':weighted,'measurements':data};rows.append(row)
                print(count,weighted,[(b,round(data[b]['submit_ms_p50'],3),round(data[b]['completed_ms_p50'],3)) for b in data],flush=True)
        finally:
            for v in viewers.values():v.close();shiboken6.delete(v);app.processEvents()
    args.output.write_text(json.dumps({'metric':'fixed orientation; warmed caches; no readback/overlay; both use native draw submission','environment':meta,'cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
