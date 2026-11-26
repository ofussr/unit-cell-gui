"""Compare sphere mesh and analytical sphere rendering, without image readback.
Synchronous completed-frame wall time with glFinish; software results do not
predict hardware throughput. No polyhedra/bonds/overlay; outlines retained.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import shiboken6
from PySide6.QtWidgets import QApplication
from OpenGL import GL
from unit_cell_gui import Atom, AtomComponent, Scene, UnitCellViewer

class Viewer(UnitCellViewer):
    def _draw_overlay(self): pass

def scene(count,radius,base):
    component=AtomComponent('c','C','C',1.,'#5588bb',radius)
    width=int(np.ceil(np.sqrt(count)))
    ix=np.arange(count); pts=np.column_stack(((ix%width-(width-1)/2)*radius*2.7,(ix//width-(width-1)/2)*radius*2.7,np.zeros(count)))
    return Scene(tuple(Atom(str(i),str(i),(component,)) for i in ix),pts,[],[],[],np.eye(3),base_radius=base)

def run(app,current,backend,samples,n):
    os.environ['UNIT_CELL_GUI_ATOM_BACKEND']=backend
    v=Viewer();v.resize(640,480);fmt=v.format();fmt.setSamples(samples);v.setFormat(fmt)
    v.set_scene(current);v.set_display_options(show_bonds=False,show_polyhedra=False,show_basis=False,show_cell=False)
    v.show();app.processEvents();v.makeCurrent();GL.glBindFramebuffer(GL.GL_FRAMEBUFFER,v.defaultFramebufferObject())
    renderer=GL.glGetString(GL.GL_RENDERER).decode(); version=GL.glGetString(GL.GL_VERSION).decode();actual=int(GL.glGetIntegerv(GL.GL_SAMPLES))
    try:
        for _ in range(3):v.paintGL();GL.glFinish()
        assert v._last_atom_backend==backend,(v._last_atom_backend,v.atom_analytic_error)
        rows=[]
        for i in range(n):
            theta=.03*i; c,s=np.cos(theta),np.sin(theta)
            v.set_orientation(np.array([[c,0,s],[0,1,0],[-s,0,c]]))
            GL.glFinish();start=time.perf_counter();v.paintGL();GL.glFinish();rows.append((time.perf_counter()-start)*1000)
        assert GL.glGetError()==GL.GL_NO_ERROR
    finally:v.doneCurrent();v.close();shiboken6.delete(v);app.processEvents()
    return {'samples_ms':rows,'p50_ms':float(np.median(rows)),'p95_ms':float(np.percentile(rows,95)),'actual_msaa':actual,'renderer':renderer,'version':version}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=10);a=p.parse_args();app=QApplication([])
    result={'metric':'completed paintGL wall ms, glFinish, no readback, software or driver as recorded','cases':[]}
    for count,r,base in [(1,.8,1.3),(16,.22,1.5),(256,.12,3.3),(1024,.06,3.3),(4096,.03,3.3)]:
        current=scene(count,r,base)
        for samples in (0,4):
            mesh=run(app,current,'mesh',samples,a.samples);glsl=run(app,current,'glsl',samples,a.samples)
            row={'atoms':count,'radius':r,'base_radius':base,'requested_msaa':samples,'mesh':mesh,'glsl':glsl,'ratio':mesh['p50_ms']/glsl['p50_ms']};result['cases'].append(row)
            print(count,samples,round(mesh['p50_ms'],3),round(glsl['p50_ms'],3),round(row['ratio'],3),flush=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
