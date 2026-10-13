from pathlib import Path
import sys
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
from unit_cell_gui import Atom, AtomComponent, DisplayOptions, Polyhedron, Scene, StyleOverrides, UnitCellViewer
from test_unit_cell_gui import sample_scene, rotation

app = QApplication([])
output = Path(sys.argv[1]); output.mkdir(exist_ok=True)
original = sample_scene()
components = tuple(AtomComponent(f'c{i}', f'C{i}', 'C', .08, f'#{i*20:02x}80b0', .2+i*.02) for i in range(10))
special = Scene((Atom('eight', 'Eight', components[:8]), Atom('ten', 'Ten', components),
                 Atom('ext', 'External', (components[0],), external=True)),
                [[-.7,.1,-.15], [.55,-.2,.13], [.1,.65,.4]], [[0,1],[1,2]],
                original.cell_vertices, original.cell_edges, original.basis_vectors, base_radius=2.)
vertices = np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
quad = Polyhedron(np.zeros(3), vertices, ((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)),
                  'centre', 'Cube', original.atoms[0].components)
cube = Scene(original.atoms, original.atom_centers, original.bonds, original.cell_vertices,
             original.cell_edges, original.basis_vectors, (quad,), original.base_radius)
cases = []
for engraving in (False,True):
    suffix = 'bw' if engraving else 'color'
    common = dict(show_polyhedra=True, engraving=engraving)
    cases.extend([
        (f'base-{suffix}', original, common, StyleOverrides(), np.eye(3)),
        (f'rotated-{suffix}', original, common, StyleOverrides(), rotation(0,27)@rotation(1,43)),
        (f'opaque-{suffix}', original, common, StyleOverrides(polyhedron_opaque_sites={'centre'}), rotation(0,17)@rotation(1,21)),
        (f'scaled-{suffix}', original, dict(common,atom_scale=1.7), StyleOverrides(), rotation(2,23)),
        (f'atoms-hidden-{suffix}', original, dict(common,show_atoms=False), StyleOverrides(), np.eye(3)),
        (f'poly-hidden-{suffix}', original, common, StyleOverrides(polyhedron_site_visibility={'centre':False}), np.eye(3)),
        (f'quad-{suffix}', cube, common, StyleOverrides(), rotation(0,19)@rotation(1,38)),
        (f'eight-ten-partial-{suffix}', special, dict(engraving=engraving), StyleOverrides(), rotation(0,35)@rotation(1,61)),
        (f'external-hidden-{suffix}', special, dict(engraving=engraving,show_external_atoms=False), StyleOverrides(), np.eye(3)),
        (f'component-hidden-{suffix}', special, dict(engraving=engraving), StyleOverrides(atom_component_visibility={'c1':False,'c5':False},atom_component_colours={'c7':'#00ff00'}), rotation(0,21)),
    ])
# Crowded atoms exercise silhouette depth occlusion in front of neighbours.
cluster_components = (AtomComponent('C', 'Carbon', 'C', 1., '#5588bb', .45),
                      AtomComponent('O', 'Oxygen', 'O', .45, '#bb8855', .45))
cluster = Scene(tuple(Atom(str(i), str(i), (cluster_components[i % 2],)) for i in range(5)),
                [[-.5,0,-.3], [0,0,.3], [.5,0,-.2], [.1,.35,-.4], [.05,-.35,.5]],
                [[0,1], [1,2]], original.cell_vertices, original.cell_edges,
                original.basis_vectors, (quad,), base_radius=2.)
for bw in (False, True):
    for angle in (0, 47):
        for opaque in (False, True):
            cases.append((f'overlap-{bw}-{angle}-{opaque}', cluster,
                          dict(engraving=bw, show_polyhedra=True),
                          StyleOverrides(polyhedron_opaque_sites={'centre'} if opaque else set()),
                          rotation(1,angle)@rotation(0,19)))
# More than one polyhedron exercises non-zero instance offsets and sorting.
def cube_scene():
    base = original
    vertices = np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
    faces = ((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
    first = Polyhedron(np.zeros(3), vertices, faces, 'cube', 'Cube', base.atoms[0].components)
    second = Polyhedron([.5,.3,-.8], vertices+[.5,.3,-.8], faces, 'other','Other',base.atoms[1].components)
    return Scene(base.atoms,base.atom_centers,base.bonds,base.cell_vertices,base.cell_edges,base.basis_vectors,(first,second),base.base_radius)
for bw in (False, True):
    for opaque in (False, True):
        cases.append((f'multiple-{bw}-{opaque}', cube_scene(),
                      dict(show_polyhedra=True, engraving=bw),
                      StyleOverrides(polyhedron_opaque_sites={'cube'} if opaque else set()),
                      rotation(0,25)@rotation(1,37)))
# Open half-cylinders in all axis directions, near the reference-frame switch.
ca=AtomComponent('a','A','C',1,'#5588bb',.3)
cb=AtomComponent('b','B','O',1,'#ff8800',.3)
for i,delta in enumerate(([2,0,0],[0,2,0],[0,0,2],[.4,1.5,-.2],[.436,0,.9],[0,0,0])):
    centers=np.array([delta],float)*np.array([[-.5],[.5]])
    bondscene=Scene((Atom('a','A',(ca,)),Atom('b','B',(cb,))),centers,[[0,1]],
                    original.cell_vertices, original.cell_edges, original.basis_vectors,base_radius=2)
    for bw in (False, True):
        cases.append((f'bond-axis-{i}-{bw}', bondscene, dict(show_atoms=False,engraving=bw),
                      StyleOverrides(),rotation(0,19)@rotation(1,31)))
# Five-sided prism: original pentagons are kept intact for hatching.
theta=np.arange(5)*2*np.pi/5
prism_vertices=np.concatenate([np.column_stack((np.cos(theta),np.sin(theta),np.full(5,z))) for z in (-.6,.6)])
prism_faces=(tuple(range(5)),tuple(range(5,10)))+tuple((i,(i+1)%5,(i+1)%5+5,i+5) for i in range(5))
prism=Polyhedron(np.zeros(3),prism_vertices,prism_faces,'prism','Prism',(ca,))
prism_scene=Scene(original.atoms,original.atom_centers,original.bonds,original.cell_vertices,
                  original.cell_edges,original.basis_vectors,(prism,),base_radius=2)
for bw in (False,True):
    cases.append((f'pentagon-{bw}',prism_scene,dict(show_polyhedra=True,engraving=bw),
                  StyleOverrides(),rotation(0,27)@rotation(1,41)))
for name,scene,options,style,orientation in cases:
    v=UnitCellViewer(); v.resize(640,480); v.set_scene(scene)
    v.set_display_options(DisplayOptions(**options)); v.set_style_overrides(style); v.set_orientation(orientation)
    v.show(); app.processEvents()
    assert v.isValid(), name
    if 'scaled' in name or 'overlap' in name:
        v.zoom_by(2)
        v.pan_x = 21; v.pan_y = -13
        v.resize(760, 540); app.processEvents()
    image=v.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
    assert v.opengl_error is None, (name,v.opengl_error)
    assert not image.isNull()
    data=np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).reshape(image.height(),image.width(),4).copy()
    np.save(output/f'{name}.npy',data); image.save(str(output/f'{name}.png')); v.close(); import shiboken6; shiboken6.delete(v); app.processEvents()
print(f'{len(cases)} frames saved to {output}')
