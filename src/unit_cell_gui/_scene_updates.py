"""Validated render-only overrides; the caller's immutable Scene is untouched."""
from collections.abc import Mapping
from dataclasses import replace
import operator
import numbers
import numpy as np
from .models import Polyhedron
from ._spatial import BoundsIndex


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
        self._atom_centers = None
        self._polyhedra = None
        self._bonds_by_atom = None
        self._bounds_index = None
        self.position_overrides = {}
        self.visibility_overrides = {}
        self.polyhedron_overrides = {}
        self.opacity_overrides = {}

    @property
    def atom_centers(self):
        return self.base.atom_centers if self._atom_centers is None else self._atom_centers

    @property
    def polyhedra(self):
        return self.base.polyhedra if self._polyhedra is None else self._polyhedra

    def writable_positions(self):
        if self._atom_centers is None:
            self._atom_centers = self.base.atom_centers.copy()
        return self._atom_centers

    def writable_polyhedra(self):
        if self._polyhedra is None:
            self._polyhedra = list(self.base.polyhedra)
        return self._polyhedra

    @property
    def bonds_by_atom(self):
        if self._bonds_by_atom is None:
            self._bonds_by_atom = [set() for _ in self.base.atoms]
            for index, (first, second) in enumerate(self.base.bonds):
                self._bonds_by_atom[int(first)].add(index)
                self._bonds_by_atom[int(second)].add(index)
        return self._bonds_by_atom

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _bounds(self, ids):
        low = np.full((len(ids), 3), np.inf)
        high = np.full((len(ids), 3), -np.inf)
        for row, index in enumerate(ids):
            vertices = self.polyhedra[index].vertices
            if len(vertices):
                low[row], high[row] = vertices.min(axis=0), vertices.max(axis=0)
        return low, high

    def update_bounds(self, indices=None):
        if self._bounds_index is not None:
            ids = list(range(len(self.polyhedra)) if indices is None else indices)
            self._bounds_index.update(ids, *self._bounds(ids))

    def sections_near(self, points_and_radii):
        if not points_and_radii or not self.polyhedra:
            return set()
        if self._bounds_index is None:
            self._bounds_index = BoundsIndex(*self._bounds(list(range(len(self.polyhedra)))))
        touched = set()
        for point, radius in points_and_radii:
            point = np.asarray(point)
            touched.update(self._bounds_index.query(point-radius, point+radius).tolist())
        return touched
