from pathlib import Path
import importlib.util
import json
import time
from statistics import median
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
import shiboken6
import unit_cell_gui
from unit_cell_gui import Atom, AtomComponent, Scene, UnitCellViewer
from unit_cell_gui import opengl_viewer as renderer

base = Path(__file__).parent
assert Path(unit_cell_gui.__file__).is_relative_to(base / 'release024-wheel-install')
app = QApplication([])
spec = importlib.util.spec_from_file_location('partial_example', base / 'unit-cell-gui/examples/partial_updates.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
host = example.Host()
host.show()
app.processEvents()
host.viewer.grabFramebuffer()
host.x.setValue(20)
host.poly.setValue(115)
host.visible.setChecked(False)
host.bw.setChecked(True)
app.processEvents()
assert not host.viewer.grabFramebuffer().isNull()
assert host.viewer.opengl_error is None
np.testing.assert_array_equal(host.viewer.get_atom_positions()[1], [1.2, 0, 0])
assert host.viewer.get_atom_visibility().tolist() == [True, False]
np.testing.assert_allclose(host.viewer.get_polyhedra()[0].vertices, host.original.polyhedra[0].vertices * 1.15)
host.reset_changes()
app.processEvents()
host.viewer.grabFramebuffer()
np.testing.assert_array_equal(host.viewer.get_atom_positions(), host.original.atom_centers)
assert host.viewer.get_atom_visibility().all()
host.close()
shiboken6.delete(host)
app.processEvents()

# A middle atom has two existing adjacent bonds. Measure actual renderer uploads
# and end-to-end framebuffer time; this includes readback and software rendering.
component = AtomComponent('C', 'C', 'C', 1, '#5599bb', .16)
count = 580
centers = np.column_stack((np.arange(count) % 29, np.arange(count) // 29, np.zeros(count))) * .5
atoms = tuple(Atom(str(i), str(i), (component,)) for i in range(count))
bonds = np.column_stack((np.arange(count - 1), np.arange(1, count)))
def scene(points):
    return Scene(atoms, points, bonds, [], [], np.eye(3), (), base_radius=9)
viewer = UnitCellViewer()
viewer.resize(640, 480)
viewer.set_scene(scene(centers))
viewer.show()
app.processEvents()
viewer.grabFramebuffer()
results = {}
for mode in ('partial', 'full'):
    viewer.set_scene(scene(centers), reset_camera=False)
    viewer.grabFramebuffer()
    samples = []
    upload_sizes = []
    allocations = []
    for step in range(10):
        updated = centers.copy()
        updated[290, 2] = .2 if step % 2 == 0 else -.2
        with patch.object(renderer, 'glBufferData', wraps=renderer.glBufferData) as data, \
             patch.object(renderer, 'glBufferSubData', wraps=renderer.glBufferSubData) as subdata:
            start = time.perf_counter()
            if mode == 'partial':
                viewer.set_atom_positions({290: updated[290]})
            else:
                viewer.set_scene(scene(updated), reset_camera=False)
            viewer.grabFramebuffer()
            elapsed = time.perf_counter() - start
        assert viewer.opengl_error is None
        if step >= 2:
            samples.append(elapsed)
            upload_sizes.append(sum(call.args[1] for call in data.call_args_list) +
                                sum(call.args[2] for call in subdata.call_args_list))
            allocations.append(data.call_count)
    results[mode] = dict(framebuffer_median_ms=median(samples)*1000,
                         renderer_upload_bytes_per_frame=upload_sizes,
                         buffer_allocations_per_frame=allocations)
viewer.close()
shiboken6.delete(viewer)
app.processEvents()
results.update(version=unit_cell_gui.__version__, installed_from=str(unit_cell_gui.__file__),
               example='passed', atoms=count, bonds=len(bonds), samples_per_mode=8,
               environment='Linux, Xvfb, Mesa software OpenGL 3.3; framebuffer readback included')
(base / 'release024-wheel-smoke.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
