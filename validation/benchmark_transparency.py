"""Measure rotation + framebuffer readback; run with the package to compare on PYTHONPATH.

Example: python validation/benchmark_transparency.py --output /tmp/timing --supersampling 1
Requires a working desktop OpenGL display. Software-renderer numbers do not
predict hardware GPU speed. The JSON includes individual measured samples.
"""
from pathlib import Path
import argparse
import json, sys, time
from statistics import median
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
import shiboken6
import unit_cell_gui
from unit_cell_gui import Atom, AtomComponent, Polyhedron, Scene, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_unit_cell_gui import rotation
from test_z_scene_instancing import cube_scene
from render_test_helpers import adjacent_cubes
from dataclasses import replace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--supersampling', type=int, choices=(1,2), default=1)
parser.add_argument('--query-batch-size', type=int, choices=(1,4), default=4,
                    help='1 is the serial-query reference used for validation')
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--frames', type=int, default=12)
args = parser.parse_args()
if args.width <= 0 or args.height <= 0 or args.frames < 4:
    parser.error('positive dimensions and at least four frames are required')
# Historical switches remain accepted for comparisons with older checkouts.
app=QApplication([])
output=args.output; output.mkdir(parents=True, exist_ok=True)
c=AtomComponent('C','C','C',1,'#4488bb',.16)
points=np.column_stack((np.arange(580)%29,np.arange(580)//29,np.zeros(580)))*.5
atom_scene=Scene(tuple(Atom(str(i),str(i),(c,)) for i in range(580)),points,
                 np.column_stack((np.arange(579),np.arange(1,580))),[],[],np.eye(3),(),base_radius=9)
faces=((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
cube=np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
colours=['#a020f0','#22bb55','#e09020']
polys=[]
for i in range(12):
    center=np.array([(i%3-1)*.65,((i//3)%2-.5)*.65,(i//6-.5)*.9])
    comp=AtomComponent(str(i),str(i),'C',1,colours[i%3],.3)
    polys.append(Polyhedron(center,cube+center,faces,str(i),str(i),(comp,)))
crowded=Scene((),[],[],[],[],np.eye(3),tuple(polys),base_radius=3)
demo_polys=[]
for key,colour,center in [('purple','#a020f0',[.2,.15,.15]),('green','#22bb55',[-.45,-.35,-.7])]:
    center=np.array(center)
    comp=AtomComponent(key,key,'C',1,colour,.3)
    demo_polys.append(Polyhedron(center,cube+center,faces,key,key,(comp,)))
demo=replace(adjacent_cubes(2),base_radius=2)
crowded=adjacent_cubes(12)
with_atoms=cube_scene()
with_atoms=replace(with_atoms,polyhedra=(
    replace(with_atoms.polyhedra[0],center=np.array([-1.3,0,0]),vertices=cube+[-1.3,0,0]),
    replace(with_atoms.polyhedra[1],center=np.array([1.3,0,0]),vertices=cube+[1.3,0,0])),base_radius=4.)
results={'version':unit_cell_gui.__version__, 'requested_supersampling':args.supersampling,
         'geometry_contract':'nonintersecting convex faces; shared boundaries permitted', 'cases':{}}
for name,current in [('atoms580',atom_scene),('polyhedra2',with_atoms),('overlap12',crowded),('demo',demo)]:
    viewer=UnitCellViewer(); viewer.resize(args.width,args.height)
    if hasattr(viewer,'transparency_supersampling'):
        viewer.set_display_options(transparency_supersampling=args.supersampling)
    viewer.set_scene(current); viewer.set_display_options(show_polyhedra=True,show_cell=False,show_basis=False)
    viewer.set_orientation(rotation(0,17)@rotation(1,29)); viewer.show(); app.processEvents()
    samples=[]; layers=[]; uploads=[]
    for step in range(args.frames):
        with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate, \
             patch.object(renderer,'glBufferSubData',wraps=renderer.glBufferSubData) as subdata:
            start=time.perf_counter()
            viewer.set_orientation(rotation(0,17)@rotation(1,29+step*.7))
            image=viewer.grabFramebuffer()
            elapsed=time.perf_counter()-start
        assert viewer.opengl_error is None,viewer.opengl_error
        if step>=2:
            samples.append(elapsed*1000)
            peeler=getattr(viewer,'_depth_peeler',None)
            layers.append(peeler.layers if peeler else 0)
            uploads.append(sum(call.args[1] for call in allocate.call_args_list)+sum(call.args[2] for call in subdata.call_args_list))
    results['cases'][name]={'framebuffer_median_ms':median(samples), 'samples_ms':samples,
        'layers':layers,'geometry_upload_bytes':uploads,
        'physical_size':[image.width(),image.height()],
        'peeling_size':list(peeler.size) if peeler and peeler.size else None,
        'peeling_texture_bytes':40*peeler.size[0]*peeler.size[1] if peeler and peeler.size else 0,
        'passes':getattr(peeler,'passes',layers[-1]+1) if peeler else 0,
        'query_batches':getattr(peeler,'query_batches',layers[-1]+1) if peeler else 0,
        'sort_conflicts':getattr(viewer,'_transparency_sort_conflicts',None)}
    from OpenGL.GL import glGetString,GL_RENDERER,GL_VERSION
    viewer.makeCurrent()
    results['renderer']=glGetString(GL_RENDERER).decode()
    results['opengl']=glGetString(GL_VERSION).decode()
    viewer.doneCurrent()
    viewer.set_orientation(rotation(0,17)@rotation(1,29))
    viewer.grabFramebuffer().save(str(output/(name+'.png')))
    viewer.close(); shiboken6.delete(viewer); app.processEvents()
(output/'timings.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
