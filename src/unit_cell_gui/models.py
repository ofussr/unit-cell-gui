"""Toolkit-neutral input types for the unit-cell renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from ._numeric import GPU_MAX, gpu_values, index_array, integer, atom_radius, scene_numeric_bounds


def _array(value, shape, dtype=float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(shape).copy()
    gpu_values(result, 'Scene coordinates')
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AtomComponent:
    """One coloured component of an already resolved atomic position."""

    key: str
    label: str
    element: str
    occupancy: float
    colour: str
    radius: float

    def __post_init__(self) -> None:
        occupancy, radius = float(self.occupancy), float(self.radius)
        if not np.isfinite(occupancy) or not np.isfinite(radius) or radius <= 0 or radius > GPU_MAX:
            raise ValueError("Component occupancy must be finite; radius must be positive, finite and float32-representable.")
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True)
class Atom:
    """One displayed site image, possibly with mixed occupancy."""

    site_key: str
    site_label: str
    components: tuple[AtomComponent, ...]
    external: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        if not self.components:
            raise ValueError("An atom must contain at least one component.")
        object.__setattr__(self, '_base_radius', atom_radius(self))


@dataclass(frozen=True)
class Polyhedron:
    """Ready world-space vertices and polygon faces around one site."""

    center: np.ndarray
    vertices: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    site_key: str
    site_label: str
    components: tuple[AtomComponent, ...]

    def __post_init__(self) -> None:
        center = _array(self.center, (3,))
        vertices = _array(self.vertices, (-1, 3))
        faces = tuple(tuple(integer(index, "face index") for index in face) for face in self.faces)
        if any(len(face) < 3 or any(index < 0 or index >= len(vertices) for index in face)
               for face in faces):
            raise ValueError("Every polyhedron face must contain valid vertex indices.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "components", tuple(self.components))

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(component.element for component in self.components)

    @property
    def coordination_number(self) -> int:
        return len(self.vertices)


@dataclass(frozen=True)
class Scene:
    """Complete render input; no file or crystallographic object is required."""

    atoms: tuple[Atom, ...]
    atom_centers: np.ndarray
    bonds: np.ndarray
    cell_vertices: np.ndarray
    cell_edges: np.ndarray
    basis_vectors: np.ndarray
    polyhedra: tuple[Polyhedron, ...] = ()
    base_radius: float = 1.0
    elements: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        atoms, polyhedra = tuple(self.atoms), tuple(self.polyhedra)
        atom_centers = _array(self.atom_centers, (-1, 3))
        bonds = index_array(self.bonds, (-1, 2), "bond index")
        cell_vertices = _array(self.cell_vertices, (-1, 3))
        cell_edges = index_array(self.cell_edges, (-1, 2), "cell-edge index")
        basis_vectors = _array(self.basis_vectors, (3, 3))
        if len(atoms) != len(atom_centers):
            raise ValueError("atoms and atom_centers must have the same length.")
        if len(bonds) and (bonds.min() < 0 or bonds.max() >= len(atoms)):
            raise ValueError("Bond indices must refer to atoms in the scene.")
        if len(cell_edges) and (cell_edges.min() < 0 or cell_edges.max() >= len(cell_vertices)):
            raise ValueError("Cell-edge indices must refer to cell vertices.")
        if np.any(np.linalg.norm(basis_vectors, axis=0) <= 1e-12):
            raise ValueError("Every basis vector must be non-zero.")
        radius = float(self.base_radius)
        if not np.isfinite(radius) or radius <= 0 or radius > GPU_MAX:
            raise ValueError("base_radius must be positive, finite and float32-representable.")
        bonds.setflags(write=False)
        cell_edges.setflags(write=False)
        elements = tuple(self.elements) or tuple(sorted({
            component.element for atom in atoms for component in atom.components
        }))
        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "atom_centers", atom_centers)
        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "cell_vertices", cell_vertices)
        object.__setattr__(self, "cell_edges", cell_edges)
        object.__setattr__(self, "basis_vectors", basis_vectors)
        object.__setattr__(self, "polyhedra", polyhedra)
        object.__setattr__(self, "base_radius", radius)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, '_numeric_bounds', scene_numeric_bounds(self))


@dataclass(frozen=True)
class DisplayOptions:
    """All caller-controlled presentation switches and numeric settings."""

    show_atoms: bool = True
    show_external_atoms: bool = True
    show_bonds: bool = True
    show_cell: bool = True
    show_basis: bool = True
    show_polyhedra: bool = False
    engraving: bool = False
    hatching: bool = True
    hatch_density: int = 22
    hatch_grip: int = 50
    atom_scale: float = 1.0
    rotation_enabled: bool = True
    transparency_supersampling: int = 1
    weighted_blended_oit_when_rotate: bool = False
    weighted_blended_oit: bool = False

    def __post_init__(self) -> None:
        density = int(self.hatch_density)
        grip = int(self.hatch_grip)
        scale = float(self.atom_scale)
        if density < 0 or grip < 0:
            raise ValueError("Hatching density and GRIP must be non-negative.")
        if not np.isfinite(scale) or scale <= 0 or scale > GPU_MAX:
            raise ValueError("atom_scale must be positive, finite and float32-representable.")
        supersampling = self.transparency_supersampling
        if (isinstance(supersampling, (bool, np.bool_))
                or not isinstance(supersampling, (int, np.integer))
                or supersampling not in (1, 2)):
            raise ValueError("transparency_supersampling must be integer 1 or 2.")
        object.__setattr__(self, "hatch_density", density)
        object.__setattr__(self, "hatch_grip", grip)
        object.__setattr__(self, "atom_scale", scale)
        object.__setattr__(self, "transparency_supersampling", int(supersampling))
        for name in ('weighted_blended_oit_when_rotate', 'weighted_blended_oit'):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool.")
            object.__setattr__(self, name, bool(getattr(self, name)))


@dataclass(frozen=True)
class StyleOverrides:
    """Per-site presentation changes supplied by the host application."""

    atom_component_visibility: Mapping[str, bool] = field(default_factory=dict)
    atom_component_colours: Mapping[str, str] = field(default_factory=dict)
    polyhedron_site_visibility: Mapping[str, bool] = field(default_factory=dict)
    polyhedron_site_colours: Mapping[str, str] = field(default_factory=dict)
    polyhedron_opaque_sites: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        visibility = self.atom_component_visibility or {}
        colours = self.atom_component_colours or {}
        poly_visibility = self.polyhedron_site_visibility or {}
        poly_colours = self.polyhedron_site_colours or {}
        object.__setattr__(self, "atom_component_visibility", MappingProxyType({
            str(key): bool(value) for key, value in visibility.items()
        }))
        object.__setattr__(self, "atom_component_colours", MappingProxyType({
            str(key): str(value) for key, value in colours.items()
        }))
        object.__setattr__(self, "polyhedron_site_visibility", MappingProxyType({
            str(key): bool(value) for key, value in poly_visibility.items()
        }))
        object.__setattr__(self, "polyhedron_site_colours", MappingProxyType({
            str(key): str(value) for key, value in poly_colours.items()
        }))
        object.__setattr__(self, "polyhedron_opaque_sites", frozenset(
            str(key) for key in self.polyhedron_opaque_sites
        ))
