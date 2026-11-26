"""Compare completed frames and command submission, no framebuffer readback.
Run against fixed a7 or a8. WBOIT backend selected before widget creation.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import shiboken6
from OpenGL import GL
from PySide6.QtWidgets import QApplication
import unit_cell_gui
from unit_cell_gui import UnitCellViewer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from render_test_helpers import adjacent_cubes
from test_unit_cell_gui import rotation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=12);args=p.parse_args()
    app=QApplication([]);rows=[];meta={}
    for count in (2,12,48,128):
        for weighted in (False,True):
            modes=('dual','auto') if weighted and unit_cell_gui.__version__=='0.3.0a8' else ('auto',)
            for mode in modes:
                os.environ['UNIT_CELL_GUI_WBOIT_BACKEND']=mode
                v=UnitCellViewer();v.resize(640,480);v.set_scene(adjacent_cubes(count));v.set_display_options(show_polyhedra=True,show_atoms=False,show_bonds=False,show_cell=False,show_basis=False,weighted_blended_oit=weighted)
                v.set_orientation(rotation(0,23)@rotation(1,37));v.show();app.processEvents();v.makeCurrent()
                assert v.opengl_error is None and v.weighted_oit_error is None,(v.opengl_error,v.weighted_oit_error)
                meta={'renderer':GL.glGetString(GL.GL_RENDERER).decode(),'version':GL.glGetString(GL.GL_VERSION).decode(),'samples':int(GL.glGetIntegerv(GL.GL_SAMPLES))}
                try:
                    def sample():
                        v._native_draw_calls=0
                        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject())
                        v._restore_render_state();GL.glClear(GL.GL_COLOR_BUFFER_BIT|GL.GL_DEPTH_BUFFER_BIT);GL.glFinish()
                        start=time.perf_counter()
                        if weighted: assert v._draw_weighted_frame(v.palette().base().color()),v.weighted_oit_error
                        else:v._draw_exact_scene()
                        submit=time.perf_counter();GL.glFinish();end=time.perf_counter()
                        return [(submit-start)*1000,(end-start)*1000,v._native_draw_calls]
                    for _ in range(3):sample()
                    values=[sample() for _ in range(args.samples)]
                    row={'polyhedra':count,'weighted':weighted,'mode':mode,'single':getattr(v._weighted_oit,'single_pass',False),'submit_ms_p50':float(np.median([x[0] for x in values])),'completed_ms_p50':float(np.median([x[1] for x in values])),'native_draws':int(np.median([x[2] for x in values])),'samples':values};rows.append(row)
                    print(row['polyhedra'],row['weighted'],row['mode'],row['single'],round(row['submit_ms_p50'],3),round(row['completed_ms_p50'],3),row['native_draws'],flush=True)
                finally:v.doneCurrent();v.close();shiboken6.delete(v);app.processEvents()
    args.output.write_text(json.dumps({'version':unit_cell_gui.__version__,'metric':'640x480, warmed, fixed view, no readback or overlay, software GL','environment':meta,'cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
