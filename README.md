# unit-cell-gui

A reusable **PySide6/OpenGL widget for crystal-structure visualization**.
The host application supplies Cartesian atoms, bonds, unit-cell geometry and
optional polyhedra; the widget handles rendering, camera controls and display
settings. CIF parsing, symmetry expansion, structure construction and diffraction
calculations belong to the host application.

- Coloured and black-and-white views, including polyhedron hatching.
- Mixed and partial atomic occupancies, per-component colours and visibility.
- Depth-tested atoms, bonds, polyhedron faces, edges and contact contours.
- External updates to individual atoms, polyhedra and opacities.
- Orientation signals for synchronization with other views.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/ofussr/unit-cell-gui/main/docs/images/color-view.png" alt="Color view" width="100%">
      <br>
      <sub>Color view</sub>
    </td>
    <td align="center" width="50%">
      <img src="https://raw.githubusercontent.com/ofussr/unit-cell-gui/main/docs/images/hatched-view.png" alt="Hatched view" width="100%">
      <br>
      <sub>Hatched view</sub>
    </td>
  </tr>
</table>

## Installation

```bash
python -m pip install unit-cell-gui
```

Requires **Python 3.10+**, NumPy, PySide6 Essentials, PyOpenGL and a working
**OpenGL 3.3 core-profile** driver. Python dependencies are installed by pip.

Import `UnitCellViewer` before creating `QApplication`, as in the example below.
This configures the shared Qt/OpenGL format for applications with multiple
viewers or displays with different DPI settings.

## Quick start

```python
import numpy as np
from PySide6.QtWidgets import QApplication
from unit_cell_gui import Atom, AtomComponent, Scene, UnitCellViewer

silicon = AtomComponent(
    key="Si:Si1", label="Si1", element="Si",
    occupancy=1.0, colour="#f0c8a0", radius=0.30,
)
scene = Scene(
    atoms=(Atom("Si:Si1", "Si1", (silicon,)),),
    atom_centers=np.array([[0.0, 0.0, 0.0]]),
    bonds=np.empty((0, 2), dtype=int),
    cell_vertices=np.array([
        [x, y, z]
        for x in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for z in (-1.0, 1.0)
    ]),
    cell_edges=np.array([
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
    ]),
    basis_vectors=np.eye(3),
    base_radius=2.0,
)

app = QApplication([])
viewer = UnitCellViewer()
viewer.set_scene(scene)
viewer.show()
app.exec()
```

`Scene` copies its arrays and makes them read-only. Coordinates and radii use the
host application's Cartesian units. Bond and edge indices must be integers;
coordinates, radii and computed display geometry must fit GPU float32 ranges.
Polyhedra are optional and hidden by default.

`UnitCellViewer`, `CrystalCanvas` and `OpenGLUnitCellViewer` refer to the same
renderer class.

## Mouse controls

| Input | Action |
| --- | --- |
| Left-button drag | Rotate |
| Right-button drag | Pan |
| Mouse wheel | Zoom |
| Shift + right button | Open transparency options |

Set `rotation_enabled=False` to disable mouse rotation while retaining pan,
zoom and programmatic orientation updates.

## Display and styling

Pass a `DisplayOptions` object or individual keywords to `set_display_options()`:

```python
from unit_cell_gui import DisplayOptions, StyleOverrides

viewer.set_display_options(DisplayOptions(show_polyhedra=True))
viewer.set_display_options(engraving=True, hatching=True, atom_scale=1.2)
viewer.set_style_overrides(StyleOverrides(
    atom_component_colours={"Si:Si1": "#55aadd"},
))
```

Display options control atoms, external atoms, bonds, the unit cell, basis
vectors, polyhedra, atom scale and engraving/hatching settings. `StyleOverrides`
provides component/site colour and visibility mappings and opaque-site choices.
The widget copies these mappings; each `set_style_overrides()` call replaces the
complete override set.

Shared GPU meshes keep geometry resident during camera changes. Hatching uses
intact polygon faces, including quadrilaterals. Contact curves follow the face
colour in colour mode and use black ink in engraving mode.

## Updating a scene externally

After `set_scene(scene)`, update selected objects without rebuilding the complete
scene. The following examples use indices and a `Polyhedron` supplied by the host:

```python
viewer.set_atom_positions({17: (x, y, z)})
viewer.set_atom_visibility({42: False})
viewer.update_polyhedra({3: updated_polyhedron})
viewer.set_polyhedron_opacities({3: 0.35})

# Apply several changes atomically.
viewer.update_scene(
    atom_positions={17: (x, y, z)},
    atom_visibility={42: False},
    polyhedra={3: updated_polyhedron},
    polyhedron_opacities={3: 0.5},
)
```

Indices refer to the original `Scene.atoms` and `Scene.polyhedra`. Positions are
absolute Cartesian coordinates. Calls run on the **GUI thread**; accepted changes
are applied during the next paint. Invalid packets leave the current state intact.
The host remains responsible for bond topology and replacement polyhedron geometry.

Opacity ranges from `0` (hidden) to `1` (opaque). `None` removes an individual
opacity override. In engraving, positive values retain opaque fills and black
ink; fractional values remain stored for colour mode.

```python
viewer.set_polyhedron_opacities({3: None})
positions = viewer.get_atom_positions()
visibility = viewer.get_atom_visibility()
polyhedra = viewer.get_polyhedra()
opacities = viewer.get_polyhedron_opacities()
viewer.reset_scene_updates()
```

Getters return independent copies of the effective state. `viewer.scene` remains
the original input. Resetting restores source geometry and removes local overrides
while preserving camera and host display/style settings. `set_scene()` clears
partial updates; `set_scene(None)` clears the view.

See the [partial-update API](docs/partial-updates.md) and
[runnable external-control example](examples/partial_updates.py).

## Orientation synchronization

Orientation is a world-to-camera matrix: `camera = orientation @ world` for a
column vector. Set it silently or request a signal explicitly:

```python
viewer.set_orientation(matrix, emit=False)
viewer.orientation_changed.connect(update_other_view)
other_view.orientation_changed.connect(
    lambda matrix: viewer.set_orientation(matrix, emit=False)
)
```

Mouse rotation emits the complete matrix through `orientation_changed`; release
emits `interaction_finished`. Silent updates support bidirectional synchronization
without a signal loop. Quality changes do not emit orientation signals.

Matrices must be finite, float32-representable and remain invertible after
float32 rounding, with condition number below `1e6`. Supported transforms include
well-conditioned nonorthogonal matrices and reflections. Invalid input leaves
the orientation unchanged.

## Transparency modes

The default renderer sorts transparent faces and physical edges together and
preserves native MSAA antialiasing. Supplied faces must be planar and convex,
without physical intersections or cyclic visibility constraints. Shared
boundaries are supported; dense overlap can make sorting expensive.

For approximate transparency, use the **Shift + right-button** menu or the API:

| Setting | Behaviour |
| --- | --- |
| Both off | Sorted transparency throughout |
| Weighted Blended OIT when rotate | Approximate during rotation; sorted after stopping |
| Weighted Blended OIT always | Approximate also at rest; takes priority when both are enabled |

```python
viewer.set_display_options(weighted_blended_oit_when_rotate=True)
viewer.set_display_options(weighted_blended_oit=True)   # permanent WBOIT
viewer.set_display_options(weighted_blended_oit=False)  # keep rotation preference
```

Both options default to false, apply independently to each widget and are not
saved to disk. Engraving ignores them. Rotation-only mode requests a sorted frame
on mouse release or after **150 ms** without an effective external
`set_orientation()` change. Pan and zoom alone do not start that mode.

WBOIT can change overlapping colours and edge emphasis, and can be slower on
small scenes. It requires additional GPU memory and compatible floating-point
MSAA targets. If unavailable, the widget uses sorted rendering and shows the
reason in the menu and `weighted_oit_error`. Its attachment budget is 256 MiB per
widget. See [transparency details and resource limits](docs/hybrid-transparency.md).

## Documentation

- [Scene models and display/style options](src/unit_cell_gui/models.py)
- [External partial updates](docs/partial-updates.md)
- [Transparency behaviour](docs/hybrid-transparency.md)
- [Validation and benchmarks](validation/README.md)
- [Changelog](CHANGELOG.md) and [release notes](RELEASE_NOTES_0.2.5.md)
- [MIT license](LICENSE)

## Used in

[**XRD Combine**](https://github.com/ofussr/XRD-Combine) uses `unit-cell-gui` for
crystal-structure visualization alongside CIF processing, diffraction tools and
synchronized crystallographic views.

## How to cite

If you use `unit-cell-gui` in academic work, please cite the Zenodo record:

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22714946-blue.svg)](https://doi.org/10.5281/zenodo.22714946)

Citation metadata is available in [`CITATION.cff`](CITATION.cff).