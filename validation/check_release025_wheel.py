from pathlib import Path
import importlib.util,json
import numpy as np
from PySide6.QtWidgets import QApplication
import shiboken6
import unit_cell_gui

base=Path(__file__).parent
assert Path(unit_cell_gui.__file__).is_relative_to(base/'release025-wheel-install')
app=QApplication([])
spec=importlib.util.spec_from_file_location('example_host',base/'unit-cell-gui/examples/partial_updates.py')
example=importlib.util.module_from_spec(spec);spec.loader.exec_module(example)
host=example.Host();host.show();app.processEvents();host.viewer.grabFramebuffer()
host.opacity.setValue(35);host.x.setValue(20);host.poly.setValue(110)
host.viewer.grabFramebuffer()
assert host.viewer.opengl_error is None
assert host.viewer.get_polyhedron_opacities()[0]==.35
np.testing.assert_array_equal(host.viewer.get_atom_positions()[1],[1.2,0,0])
host.opacity.setValue(100);host.viewer.grabFramebuffer()
assert host.viewer.get_polyhedron_opacities()[0]==1
host.opacity.setValue(0);host.viewer.grabFramebuffer()
assert host.viewer.get_polyhedron_opacities()[0]==0
host.reset_changes();host.viewer.grabFramebuffer()
assert host.viewer.get_polyhedron_opacities()[0]==.64
np.testing.assert_array_equal(host.viewer.get_atom_positions(),host.original.atom_centers)
host.close();shiboken6.delete(host);app.processEvents()
result={'version':unit_cell_gui.__version__,'installed_wheel':'passed','example_opacity_slider':'passed',
        'zero_opaque_fractional_and_reset':'passed','external_geometry_controls':'passed'}
(base/'release025-wheel-smoke.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
