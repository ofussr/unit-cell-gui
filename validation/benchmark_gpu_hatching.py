"""CPU preparation and buffer payload benchmark; not a GPU FPS benchmark."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_visibility_cache import repeated_scene
from test_unit_cell_gui import rotation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=12);args=p.parse_args()
    app=QApplication([]);rows=[]
    for count in (1,5,20,64):
        viewer=UnitCellViewer();viewer.resize(800,600);viewer.set_scene(repeated_scene(count));viewer.set_display_options(engraving=True,show_polyhedra=True)
        viewer.set_orientation(rotation(0,23)@rotation(1,37));plan=viewer._hatch_plan()
        methods={'python':lambda: [viewer._build_polyhedron_hatch_geometry(i,d) for i,d in plan],
                 'glsl':lambda: [viewer._gpu_hatch_plan(i,d)[0] for i,d in plan]}
        samples={k:[] for k in methods};sizes={k:sum(a.nbytes for a in f()) for k,f in methods.items()}
        for j in range(args.samples):
            for k in (('python','glsl') if j%2==0 else ('glsl','python')):
                t=time.perf_counter();methods[k]();samples[k].append((time.perf_counter()-t)*1000)
        rows.append(dict(polyhedra=count,bytes=sizes,samples_ms=samples,p50_ms={k:float(np.median(v)) for k,v in samples.items()},p95_ms={k:float(np.percentile(v,95)) for k,v in samples.items()}));print(rows[-1]['polyhedra'],rows[-1]['bytes'],rows[-1]['p50_ms'],flush=True)
        viewer.close()
    args.output.write_text(json.dumps({'metric':'CPU uncached hatch preparation only, no GL calls; bytes are GPU input payload, not measured bandwidth','cases':rows},indent=2)+'\n')
if __name__=='__main__':main()
