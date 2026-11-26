"""CPU contact geometry comparison in the same wheel, excluding OpenGL."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import shiboken6
from PySide6.QtWidgets import QApplication
from unit_cell_gui import UnitCellViewer
from benchmark_updates import contact_scene

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen');app=QApplication([])
    results={}
    for count in (1,16,64,256):
        scene=contact_scene(count);v=UnitCellViewer();v.set_scene(scene)
        row={}
        for cold in (False,True):
            mode='cold' if cold else 'warm';samples={k:[] for k in ('python','rust')};vertices={}
            for i in range(10):
                for backend in (('python','rust') if i%2==0 else ('rust','python')):
                    os.environ['UNIT_CELL_GUI_CONTACT_BACKEND']=backend
                    if cold: v._contact_atoms=None;v._contact_face_cache.clear()
                    start=time.perf_counter();records,ranges=v._build_atom_face_intersection_geometry();elapsed=(time.perf_counter()-start)*1000
                    if i: samples[backend].append(elapsed)
                    vertices[backend]=len(records)
            row[mode]={k:{'samples_ms':s,'p50_ms':float(np.median(s)),'p95_ms':float(np.percentile(s,95)), 'vertices':vertices[k]} for k,s in samples.items()}
        results[str(count)]=row
        print(count,[(mode,round(row[mode]['python']['p50_ms']/row[mode]['rust']['p50_ms'],2)) for mode in row],flush=True)
        v.close();shiboken6.delete(v);app.processEvents()
    args.output.write_text(json.dumps({'metric':'CPU contact geometry; 9 timed repetitions per backend, alternating order; no GL','cases':results},indent=2))
if __name__=='__main__':main()
