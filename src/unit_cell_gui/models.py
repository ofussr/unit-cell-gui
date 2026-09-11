"""Toolkit-neutral input types for the unit-cell renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _array(value, shape, dtype=float) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(shape).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("Scene coordinates must be finite.")
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
        if not np.isfinite(occupancy) or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Component occupancy and radius must be finite; radius must be positive.")
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
        faces = tuple(tuple(int(index) for index in face) for face in self.faces)
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
        bonds = np.asarray(self.bonds, dtype=int).reshape(-1, 2).copy()
        cell_vertices = _array(self.cell_vertices, (-1, 3))
        cell_edges = np.asarray(self.cell_edges, dtype=int).reshape(-1, 2).copy()
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
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("base_radius must be finite and positive.")
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

    def __post_init__(self) -> None:
        density = int(self.hatch_density)
        grip = int(self.hatch_grip)
        scale = float(self.atom_scale)
        if density < 0 or grip < 0:
            raise ValueError("Hatching density and GRIP must be non-negative.")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("atom_scale must be finite and positive.")
        object.__setattr__(self, "hatch_density", density)
        object.__setattr__(self, "hatch_grip", grip)
        object.__setattr__(self, "atom_scale", scale)


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
