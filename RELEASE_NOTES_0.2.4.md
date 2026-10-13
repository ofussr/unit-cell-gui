# unit-cell-gui 0.2.4

Changes relative to 0.2.3. This release extends shared GPU geometry to atom
outlines, bonds and polyhedra, adds external partial scene updates, and fixes
the dark contact curves between coloured atoms and polyhedron faces.
Existing host applications can keep using their current integration code.

## Rendering and contact colours

- Atom silhouettes use one shared camera-facing 64-segment ring and compact
  instance records. Rotation and zoom update uniforms instead of rebuilding
  contour vertices on the CPU. Partial occupancy in engraving mode keeps
  the original dashed contours.
- Coloured bonds use one shared 12-segment cylinder, with two instance records
  per bond. Endpoint colours, midpoint joins, lighting and thickness are retained.
- Polyhedron fill triangles, physical edges and engraving bonds use shared
  primitives with resident instance records. Transparency order is preserved;
  depth-sorted edges use compact index lists. Large texture-buffer arrays are
  split according to the actual driver's storage limit.
- Atom/face intersection curves now use the corresponding face's colour in
  coloured mode, including host polyhedron colour overrides. They no longer
  produce artificial black seams at contacts. Engraving mode retains black
  intersection ink. Colours refresh correctly after mode/style changes and
  partial polyhedron replacement.
- Atom tessellation remains 720 triangles per sphere. Line widths, depth
  offsets, antialiasing, occupancy sectors and the intact-polygon hatching
  algorithm are retained. This release does not add sphere LOD.

## External partial updates

```python
viewer.update_scene(
    atom_positions={17: (x, y, z)},
    atom_visibility={42: False},
    polyhedra={3: updated_polyhedron},
)
```

Indices refer to the original Scene arrays; positions are absolute Cartesian
coordinates. Packets are validated atomically, copied and merged before the
next paint. Only supplied indices change. Call these methods on the GUI thread.

The individual setters are `set_atom_positions`, `set_atom_visibility` and
`update_polyhedra`. `reset_scene_updates()` restores the original geometry
and removes local hiding. `get_atom_positions()`, `get_atom_visibility()` and
`get_polyhedra()` return independent copies of the effective render state.

- Atom movement updates only its sphere/outline records and existing adjacent
  bonds. Hiding also hides attached bonds and intersection curves.
- Replaced polyhedra have independent fill/edge buffers. Topology changes
  resize only that object's storage. Dependent intersections and hatch caches
  refresh as needed; other resident geometry is reused.
- The original `viewer.scene` remains unchanged. Camera, zoom, pan and host
  style restrictions are preserved. `set_scene()` clears partial overrides.
- The host remains responsible for bond topology and updated polyhedra; these
  are not inferred automatically. Every accepted change requests a new frame,
  without requiring a complete scene rebuild.

See [the API contract](docs/partial-updates.md) and
[the external-control example](examples/partial_updates.py).

## Compatibility

Existing viewer aliases, Scene/Atom/Polyhedron models, method signatures,
signals and dependencies are retained. New methods are additive. OpenGL 3.3
core and GLSL 330 remain sufficient; no native extension is introduced.
GPU resources are cleaned up on scene/context teardown and reconstructed
after context recreation, including partial render state.

## Validation

The source and installed wheel are checked with real Mesa software OpenGL,
including colour/BW mode changes, contact colour overrides, partial updates,
full-scene reference images, transparency, buffer limits and context cleanup.
Detailed results are supplied in `VALIDATION_0.2.4.md` in the release archive.
Windows and hardware GPU drivers are not tested in this environment; rendering
speed depends on the scene and driver.
