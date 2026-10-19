"""Cold opacity, contact preparation, local atom updates and first-scene frames.

Run separately with each package on PYTHONPATH; scenes and measurements are
identical. --gl requires a working desktop OpenGL display. CPU phases exclude
OpenGL and framebuffer readback. Memory sampling is a separate tracemalloc pass.
"""
import argparse
import json
import os
from pathlib import Path
import time
import tracemalloc
from unittest.mock import patch
import numpy as np
import shiboken6
import unit_cell_gui
from unit_cell_gui import Atom,AtomComponent,Polyhedron,Scene,UnitCellViewer,StyleOverrides
from unit_cell_gui import opengl_viewer as renderer
from PySide6.QtWidgets import QApplication


def contact_scene(count):
    corners=np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
    faces=((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
    offsets=np.array([[.8,0,0],[-.8,0,0],[0,.8,0],[0,-.8,0],[0,0,.8],[0,0,-.8],[0,0,0],[0,0,.3]])
    component=AtomComponent('c','C','C',1,'#5588bb',.22)
    atom=Atom('a','A',(component,));polys=[];centres=[]
    for index in range(count):
        centre=np.array([(index%32)*3.,(index//32)*3.,0.])
        polys.append(Polyhedron(centre,corners+centre,faces,f'p{index}',f'P{index}',(component,)))
        centres.extend(centre+offsets)
    centres=np.asarray(centres)
    bonds=np.column_stack((np.arange(len(centres)-1),np.arange(1,len(centres))))
    return Scene((atom,)*len(centres),centres,bonds,[],[],np.eye(3),tuple(polys),base_radius=max(3.,float(np.max(np.abs(centres)))+2.))


def measured(function, count):
    values=[]
    for _ in range(count):
        start=time.perf_counter();function();values.append((time.perf_counter()-start)*1000)
    return {'samples_ms':values,'p50_ms':float(np.percentile(values,50)),
            'p95_ms':float(np.percentile(values,95))}


def dispose(widget,app):
    widget.close();shiboken6.delete(widget);app.processEvents()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gl',action='store_true')
    args=parser.parse_args()
    if not args.gl:os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    app=QApplication([])
    data={'version':unit_cell_gui.__version__,'platform':app.platformName(),'cases':{}}
    if args.gl:
        if app.platformName() in {'offscreen','minimal'}:raise RuntimeError('OpenGL display required')
        for count in (16,64):
            scene=contact_scene(count);first=[];moves=[];uploads=[]
            for trial in range(3):
                viewer=UnitCellViewer();viewer.resize(640,480)
                viewer.set_display_options(show_polyhedra=True,show_cell=False,show_basis=False,show_bonds=False)
                viewer.show();app.processEvents()  # context/shader initialization excluded
                start=time.perf_counter();viewer.set_scene(scene);viewer.grabFramebuffer()
                first.append((time.perf_counter()-start)*1000)
                viewer.set_atom_positions({1:scene.atom_centers[1]+[.03,.02,0]});viewer.grabFramebuffer()
                for step in range(4):
                    point=scene.atom_centers[1]+[.04+.01*(step%2),.02,0]
                    start=time.perf_counter();viewer.set_atom_positions({1:point});viewer.grabFramebuffer()
                    moves.append((time.perf_counter()-start)*1000)
                with patch.object(renderer,'glBufferData',wraps=renderer.glBufferData) as allocate,                      patch.object(renderer,'glBufferSubData',wraps=renderer.glBufferSubData) as update:
                    viewer.set_style_overrides(StyleOverrides(polyhedron_site_colours={'p0':'#ff0088'}))
                    viewer.grabFramebuffer()
                    uploads.append(sum(call.args[1] for call in allocate.call_args_list)
                                   +sum(call.args[2] for call in update.call_args_list))
                assert viewer.opengl_error is None,viewer.opengl_error
                dispose(viewer,app)
            data['cases'][str(count)]={'polyhedra':count,'atoms':len(scene.atoms),
                'one_site_style_gpu_uploaded_bytes':uploads,
                'first_scene_frame':{'samples_ms':first,'p50_ms':float(np.median(first))},
                'warm_atom_move_frame':{'samples_ms':moves,'p50_ms':float(np.median(moves)),
                                        'p95_ms':float(np.percentile(moves,95))}}
            print('gl',count,data['cases'][str(count)]['first_scene_frame']['p50_ms'],flush=True)
        data['metric']='set_scene + first loaded frame, and warm atom setter + frame; includes GPU and readback, excludes context initialization'
    else:
        for count in (64,256,1024):
            start=time.perf_counter();scene=contact_scene(count);construction=(time.perf_counter()-start)*1000
            samples=[];scene_times=[]
            for _ in range(15):
                viewer=UnitCellViewer()
                start=time.perf_counter();viewer.set_scene(scene);scene_times.append((time.perf_counter()-start)*1000)
                start=time.perf_counter();viewer.set_polyhedron_opacities({0:.35});samples.append((time.perf_counter()-start)*1000)
                dispose(viewer,app)
            viewer=UnitCellViewer();viewer.set_scene(scene)
            tracemalloc.start();viewer.set_polyhedron_opacities({0:.35});_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            dispose(viewer,app)
            data['cases'][f'cold_opacity_{count}']={'polyhedra':count,'atoms':len(scene.atoms),
                'samples_ms':samples,'p50_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),
                'separate_peak_traced_bytes':peak,'set_scene_samples_ms':scene_times,
                'scene_construction_ms':construction}
            print('opacity',count,float(np.median(samples)),peak,flush=True)
        for count in (16,64,256):
            scene=contact_scene(count);viewer=UnitCellViewer();viewer.set_scene(scene)
            viewer.set_display_options(show_polyhedra=True)
            def cold_contacts():
                if hasattr(viewer,'_contact_atoms'):viewer._contact_atoms=None
                if hasattr(viewer,'_contact_face_cache'):viewer._contact_face_cache.clear()
                viewer._build_atom_face_intersection_geometry()
            cold=measured(cold_contacts,5)
            # Warm the geometry/index structures before local-update timings.
            viewer.set_atom_positions({1:scene.atom_centers[1]+[.01,.02,0]})
            viewer._build_atom_face_intersection_geometry();viewer._pending_sections.clear()
            step=0
            def local_move():
                nonlocal step
                step+=1
                viewer.set_atom_positions({1:scene.atom_centers[1]+[.04+.01*(step%2),.02,0]})
                for index in sorted(viewer._pending_sections):
                    viewer._build_atom_face_intersection_geometry(poly_indices=(index,))
                viewer._pending_sections.clear()
            warm=measured(local_move,15)
            data['cases'][f'contacts_{count}']={'polyhedra':count,'atoms':len(scene.atoms),
                                             'cold_contact_cpu':cold,'warm_atom_and_contacts_cpu':warm}
            print('contacts',count,cold['p50_ms'],warm['p50_ms'],flush=True)
            dispose(viewer,app)
        data['metric']='CPU setters/contact geometry only; no OpenGL or readback; set_scene timing is separate'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
