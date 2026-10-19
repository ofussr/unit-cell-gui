import numpy as np
from unit_cell_gui import AtomComponent, Polyhedron, Scene


def face(key, colour, depth, low=-1, high=1):
    corners = np.array([[-1,low,0],[1,low,0],[1,high,0],[-1,high,0]],float)
    corners[:,2] = depth(corners[:,0]) if callable(depth) else depth
    component = AtomComponent(key,key,'C',1,colour,.3)
    return Polyhedron(corners.mean(axis=0)-[0,0,.2],corners,((0,1,2,3),),key,key,(component,))


def scene(polys):
    return Scene((),[],[],[],[],np.eye(3),tuple(polys),base_radius=2)


def adjacent_cubes(count=12):
    corners=np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
    faces=((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
    centres=np.array([[(i%3)*1.6,((i//3)%2)*1.6,(i//6)*1.6] for i in range(count)])
    centres-=centres.mean(axis=0)
    polys=[]
    for i,centre in enumerate(centres):
        key=f'cube-{i:03d}'
        component=AtomComponent(key,key,'C',1,('#a020f0','#22bb55','#e09020')[i%3],.3)
        polys.append(Polyhedron(centre,corners+centre,faces,key,key,(component,)))
    return Scene((),[],[],[],[],np.eye(3),tuple(polys),base_radius=4)
