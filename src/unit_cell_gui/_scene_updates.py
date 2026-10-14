"""Validated render-only overrides; the caller's immutable Scene is untouched."""
from collections.abc import Mapping
from dataclasses import replace
import operator
import numbers
import numpy as np
from .models import Polyhedron


def index_key(key, length, kind):
    if isinstance(key, (bool, np.bool_)):
        raise TypeError(f'{kind} index must be an integer, not bool')
    try:
        index = operator.index(key)
    except TypeError as exc:
        raise TypeError(f'{kind} index must be an integer') from exc
    if not 0 <= index < length:
        raise IndexError(f'{kind} index {index} outside [0, {length})')
    return index


def same_polyhedron(first, second):
    return (first.site_key == second.site_key and first.site_label == second.site_label
            and first.components == second.components and first.faces == second.faces
            and np.array_equal(first.center, second.center)
            and np.array_equal(first.vertices, second.vertices))


def prepare(scene, atom_positions, atom_visibility, polyhedra, polyhedron_opacities=None):
    result = []
    for mapping, kind, length in ((atom_positions, 'atom', len(scene.atoms)),
                                   (atom_visibility, 'atom', len(scene.atoms)),
                                   (polyhedra, 'polyhedron', len(scene.polyhedra)),
                                   (polyhedron_opacities, 'polyhedron', len(scene.polyhedra))):
        if mapping is None:
            result.append({}); continue
        if not isinstance(mapping, Mapping):
            raise TypeError('partial updates must be mappings indexed by scene indices')
        result.append({index_key(key, length, kind): value for key, value in mapping.items()})
    positions, visibility, polys, opacities = result
    limit = np.finfo(np.float32).max
    for index, value in list(positions.items()):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(np.abs(value) > limit):
            raise ValueError('atom positions must be three finite float32-representable Cartesian coordinates')
        positions[index] = value.copy()
    for index, value in list(visibility.items()):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError('atom visibility values must be bool')
        visibility[index] = bool(value)
    for index, value in list(polys.items()):
        if not isinstance(value, Polyhedron):
            raise TypeError('polyhedron updates must contain Polyhedron objects')
        if np.any(np.abs(value.center) > limit) or np.any(np.abs(value.vertices) > limit):
            raise ValueError('polyhedron coordinates must be float32-representable')
        polys[index] = replace(value)  # independently own all coordinate arrays
    for index, value in list(opacities.items()):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TypeError('polyhedron opacity must be a real number in [0, 1] or None')
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('polyhedron opacity must be finite and in [0, 1]')
        opacities[index] = value
    return positions, visibility, polys, opacities


class SceneView:
    def __init__(self, base):
        self.base = base
        self.atom_centers = base.atom_centers.copy()
        self.polyhedra = list(base.polyhedra)
        self.position_overrides = {}
        self.visibility_overrides = {}
        self.polyhedron_overrides = {}
        self.opacity_overrides = {}
        self.bonds_by_atom = [set() for _ in base.atoms]
        for index, (first, second) in enumerate(base.bonds):
            self.bonds_by_atom[int(first)].add(index)
            self.bonds_by_atom[int(second)].add(index)
        self.bounds = [None] * len(self.polyhedra)
        self.update_bounds()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def update_bounds(self, indices=None):
        for index in (range(len(self.polyhedra)) if indices is None else indices):
            poly = self.polyhedra[index]
            self.bounds[index] = (poly.vertices.min(axis=0), poly.vertices.max(axis=0)) if len(poly.vertices) else None

    def sections_near(self, points_and_radii):
        if not points_and_radii:
            return set()
        ids = [i for i, bounds in enumerate(self.bounds) if bounds is not None]
        if not ids:
            return set()
        low = np.asarray([self.bounds[i][0] for i in ids])
        high = np.asarray([self.bounds[i][1] for i in ids])
        points = np.asarray([item[0] for item in points_and_radii])
        radii = np.asarray([item[1] for item in points_and_radii])
        touched = np.zeros(len(ids), dtype=bool)
        for first in range(0, len(points), 128):
            block = points[first:first+128]
            radius = radii[first:first+128][None, :, None]
            overlap = np.all(block[None, :, :] >= low[:, None, :] - radius, axis=2)
            overlap &= np.all(block[None, :, :] <= high[:, None, :] + radius, axis=2)
            touched |= np.any(overlap, axis=1)
        return {ids[i] for i in np.flatnonzero(touched)}
