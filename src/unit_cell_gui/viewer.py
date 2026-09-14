"""Compatibility facade for the public unit-cell viewer.

``UnitCellViewer`` and ``CrystalCanvas`` now use the OpenGL renderer, while
keeping the import path and external API used by existing applications.
"""

from __future__ import annotations

from .opengl_viewer import OpenGLUnitCellViewer
from .view_helpers import _polyhedron_hatches, screen_drag_orientation

# Preserve every existing external construction/import call:
#   from unit_cell_gui import UnitCellViewer
#   from unit_cell_gui.viewer import UnitCellViewer
#   CrystalCanvas(...)
UnitCellViewer = OpenGLUnitCellViewer
CrystalCanvas = UnitCellViewer

__all__ = [
    "CrystalCanvas",
    "UnitCellViewer",
    "screen_drag_orientation",
]
