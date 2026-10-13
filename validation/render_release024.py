"""Shared comparison cases with optional restoration of the old contact colour."""
import runpy
import sys
from pathlib import Path
from unit_cell_gui import UnitCellViewer

if len(sys.argv) > 2 and sys.argv[2] == 'old-contact-colour':
    del sys.argv[2]
    original = UnitCellViewer._build_atom_face_intersection_geometry
    def old_contact_colour(self, **kwargs):
        data, ranges = original(self, **kwargs)
        if not self.engraving:
            data[:, 6:9] = 20 / 255
        return data, ranges
    UnitCellViewer._build_atom_face_intersection_geometry = old_contact_colour
runpy.run_path(str(Path(__file__).with_name('render_instancing02.py')), run_name='__main__')
