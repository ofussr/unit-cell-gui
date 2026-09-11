"""A Qt widget that renders complete, caller-supplied unit-cell scenes."""

from .hatching import (
    face_hatch_segments,
    hatch_division_count,
    occupancy_fractions,
    propagated_hatch_directions,
    quadrilateral_hatch_line_count,
)
from .models import Atom, AtomComponent, DisplayOptions, Polyhedron, Scene, StyleOverrides
__version__ = "0.1.0"


def __getattr__(name):
    if name in {"CrystalCanvas", "UnitCellViewer", "screen_drag_orientation"}:
        from .viewer import CrystalCanvas, UnitCellViewer, screen_drag_orientation
        return {
            "CrystalCanvas": CrystalCanvas,
            "UnitCellViewer": UnitCellViewer,
            "screen_drag_orientation": screen_drag_orientation,
        }[name]
    raise AttributeError(name)

__all__ = [
    "Atom",
    "AtomComponent",
    "CrystalCanvas",
    "DisplayOptions",
    "Polyhedron",
    "Scene",
    "StyleOverrides",
    "UnitCellViewer",
    "face_hatch_segments",
    "hatch_division_count",
    "occupancy_fractions",
    "propagated_hatch_directions",
    "quadrilateral_hatch_line_count",
    "screen_drag_orientation",
]
