from pathlib import Path
import json, sys, time
from statistics import median
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
import shiboken6
import unit_cell_gui
from unit_cell_gui import Atom, AtomComponent, Polyhedron, Scene, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer
from test_unit_cell_gui import rotation
from test_z_scene_instancing import cube_scene

app=QApplication([])
output=Path(sys.argv[1]); output.mkdir(exist_ok=True)
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
demo=Scene((),[],[],[],[],np.eye(3),tuple(demo_polys),base_radius=1.7)
results={'version':unit_cell_gui.__version__,'cases':{}}
for name,current in [('atoms580',atom_scene),('polyhedra2',cube_scene()),('overlap12',crowded),('demo',demo)]:
    viewer=UnitCellViewer(); viewer.resize(640,480)
    viewer.set_scene(current); viewer.set_display_options(show_polyhedra=True,show_cell=False,show_basis=False)
    viewer.set_orientation(rotation(0,17)@rotation(1,29)); viewer.show(); app.processEvents()
    samples=[]; layers=[]; uploads=[]
    for step in range(10):
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
    results['cases'][name]={'framebuffer_median_ms':median(samples),'layers':layers,'geometry_upload_bytes':uploads}
    viewer.set_orientation(rotation(0,17)@rotation(1,29))
    viewer.grabFramebuffer().save(str(output/(name+'.png')))
    viewer.close(); shiboken6.delete(viewer); app.processEvents()
(output/'timings.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
