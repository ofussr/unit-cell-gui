"""Reusable atom bounds and circle sampling for finite-face contact curves."""
import math
import numpy as np
from ._spatial import BoundsIndex

_ANGLES = [2.*math.pi*index/160 for index in range(160)]
CIRCLE_COS = np.asarray([math.cos(angle) for angle in _ANGLES])[:, None]
CIRCLE_SIN = np.asarray([math.sin(angle) for angle in _ANGLES])[:, None]


class ContactAtoms:
    def __init__(self, scene, radius, visible):
        self.centres = np.asarray(scene.atom_centers, dtype=float).copy()
        self.radii = np.asarray([radius(atom) for atom in scene.atoms])
        self.visible = np.asarray([visible(i) for i in range(len(scene.atoms))], dtype=bool)
        self.visible_count = int(np.count_nonzero(self.visible))
        self.index = BoundsIndex(*self._bounds(np.arange(len(scene.atoms))))

    def _bounds(self, ids):
        low = np.nextafter(self.centres[ids] - self.radii[ids, None], -np.inf)
        high = np.nextafter(self.centres[ids] + self.radii[ids, None], np.inf)
        hidden = ~self.visible[ids]
        low[hidden], high[hidden] = np.inf, -np.inf
        return low, high

    def update(self, ids, scene, radius, visible):
        ids = np.asarray(sorted(ids), dtype=int)
        if not len(ids):
            return
        self.centres[ids] = scene.atom_centers[ids]
        self.radii[ids] = [radius(scene.atoms[i]) for i in ids]
        previous = int(np.count_nonzero(self.visible[ids]))
        self.visible[ids] = [visible(int(i)) for i in ids]
        self.visible_count += int(np.count_nonzero(self.visible[ids])) - previous
        self.index.update(ids, *self._bounds(ids))

    def near_face(self, low, high):
        return self.index.query(low, high)


class ContactFace:
    def __init__(self, polygon, normal, u, v):
        self.polygon,self.normal,self.u,self.v=polygon,normal,u,v
        self.low,self.high=polygon.min(axis=0),polygon.max(axis=0)
        edges=np.roll(polygon,-1,axis=0)-polygon
        self.side_normals=np.cross(normal,edges)
        self.tolerance=1e-9*max(1.,float(np.max(np.linalg.norm(polygon-polygon[0],axis=1))))

    def contains(self, points):
        # The scalar triple product equals cross(edge, point-origin) dot normal.
        # Cache its constant coefficients; subtract origins before evaluation
        # to retain precision for translated geometry.
        sides=np.einsum('ijk,jk->ij',points[:,None,:]-self.polygon[None,:,:],self.side_normals)
        return np.all(sides>=-self.tolerance,axis=1)|np.all(sides<=self.tolerance,axis=1)


def contact_faces(polyhedron):
    result=[]
    for face in polyhedron.faces:
        polygon=polyhedron.vertices[list(face)]
        normal=np.cross(polygon[1]-polygon[0],polygon[2]-polygon[0])
        length=float(np.linalg.norm(normal))
        if length<=1e-12:
            continue
        normal/=length
        if float(np.dot(normal,polygon.mean(axis=0)-polyhedron.center))<0.:
            normal=-normal
        u=polygon[1]-polygon[0]
        u-=float(np.dot(u,normal))*normal
        length=float(np.linalg.norm(u))
        if length<=1e-12:
            continue
        u/=length
        v=np.cross(normal,u)
        length=float(np.linalg.norm(v))
        if length<=1e-12:
            continue
        v/=length
        result.append(ContactFace(polygon,normal,u,v))
    return tuple(result)
