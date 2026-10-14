# External partial scene updates (0.2.5)

## API

```python
viewer.update_scene(
    atom_positions={17: (x, y, z), 42: (x2, y2, z2)},
    atom_visibility={17: False, 42: True},
    polyhedra={3: updated_polyhedron},
    polyhedron_opacities={3: 0.35},
)
```

Indices refer to the original `Scene.atoms` and `Scene.polyhedra`. A shared
site/component key does not merge atom instances. Only supplied indices
change; successive calls merge their overrides. Indices remain stable until
`set_scene()` replaces the scene.

Positions are absolute Cartesian coordinates, using the same units as the
original scene. Values must be three finite float32-representable numbers.
Visibility values must be `bool`. Polyhedron values must be `Polyhedron` objects;
coordinates are copied. Invalid indices, types or coordinates reject the
entire packet before any change is committed. No-op packets request no repaint.

Convenience methods use the same packet validation:

```python
viewer.set_atom_positions({17: (x, y, z)})
viewer.set_atom_visibility({17: False})
viewer.update_polyhedra({3: updated_polyhedron})
viewer.set_polyhedron_opacities({3: 0.35, 7: 1.0})
viewer.set_polyhedron_opacities({3: None})  # restore this object's default
viewer.reset_scene_updates()

positions = viewer.get_atom_positions()  # independent NumPy copy
visibility = viewer.get_atom_visibility()  # effective bool mask
polyhedra = viewer.get_polyhedra()  # independently owned model copies
opacities = viewer.get_polyhedron_opacities()  # independent NumPy copy
```

`viewer.scene` remains the original input object. Getters return the effective
render state; visibility includes the host's global, component and external-atom
switches. Local `True` cannot override hiding configured by the host.
Reset restores original positions/polyhedra and removes local hiding/opacities. It
preserves camera, zoom, pan and host display/style settings. `set_scene()` clears
all partial overrides, including when called with the same Scene instance.

Call these methods on the GUI thread, like the existing viewer methods.
Accepted packets request one asynchronous repaint. GPU changes are applied
with the widget's context current during the next paint; multiple packets
before that paint coalesce. A loaded Scene is required for indexed writes.
Read getters return empty arrays/tuples when no scene is loaded.

## Polyhedron opacity

Opacity values must be finite real scalars in [0, 1]; booleans and strings
are rejected. 0 hides the polyhedron; 1 makes it opaque. `None` removes only
the supplied index's override. Indices remain tied to the original Scene;
sharing a site key does not merge different polyhedron instances.

An explicit value pins that object's opacity, even when it equals the current
default. Subsequent geometry/component/style changes retain the override.
Repeating the same stored value is a no-op. Opacity can be included in the
same atomic packet as atom/polyhedron geometry changes.

In coloured mode, an explicit index value overrides occupancy-derived opacity
and `polyhedron_opaque_sites` styling. Site/global visibility still restricts
what is drawn. Without a local override, the default is 1 for an opaque site,
otherwise `0.28 + 0.36 * min(1, summed occupancy)`.

Engraving retains opaque white faces and black ink for positive values. Zero
hides faces, edges, hatches and contact curves; stored fractional values apply
again when returning to colour mode. The opacity getter reports effective
material values for the current mode, independently of visibility switches.

The separate R32F GPU table holds one float per original polyhedron. Only
changed float slots are uploaded; adjacent changes coalesce. Crossing the
0/1 boundaries changes which resident ranges are submitted, without rebuilding
meshes or reallocating geometry buffers. Fill and physical-edge opacity change
together. Explicit overrides also fade contact curves; unset defaults preserve
their prior appearance. Camera and input Scene remain unchanged.

## Update behaviour

- Atom spheres and outlines keep stable GPU slots under local hiding; hidden
  slots are clipped in the vertex shader. Unchanged records are not regenerated.
- Movement updates only the target atom records and existing adjacent bonds.
  Hiding removes its sphere, outline, intersection ink and attached bonds.
- Bonds keep two stable half-cylinder records, including clipped zero-length
  slots, so a collapsed bond can become visible without reallocating the layer.
- Replaced polyhedra use independent fill/edge buffers. Changed counts/topology
  resize only the affected object's buffers; unchanged topology uses subdata.
  Other polyhedra and the cell keep their original resident geometry.
- Finite sphere/face intersection ink is refreshed only for polyhedra near the
  old or new affected spheres, or for replaced polyhedra. Layers that are hidden
  defer section work until enabled.
- Hatching retains the existing intact-polygon algorithm and line draw pass.
  A replaced polyhedron invalidates its own geometry cache. If moving its center
  changes global depth-based density normalization, other dependent density
  variants are also refreshed as required.
- Logical line widths and depth offsets remain unchanged. Transparent face/edge
  order is resolved per pixel using depth peeling, with supersampled antialiasing.
  Coloured atom/face contact curves match the effective face colour;
  engraving retains black intersection ink. Dynamic per-object resources are released on scene clear, host
  full-cache refresh and context cleanup, and recreated after context loss.

The package does not infer new bonds, move polyhedra automatically with atoms,
change the lattice, refit the camera or mutate caller-owned data. The host
supplies the updated polyhedron explicitly and remains responsible for topology.
The original public API, models, aliases, signals and dependencies are retained.

## Installation and example

```powershell
py -m pip install --force-reinstall --no-deps ".\dist\unit_cell_gui-0.2.5-py3-none-any.whl"
py -c "import unit_cell_gui; print(unit_cell_gui.__version__)"
py .\examples\partial_updates.py
```

Use the application's Python environment and restart the application after
installation. The example's external controls call the public batch API.
See `RELEASE_NOTES_0.2.5.md` for changes since 0.2.4. The release archive also
contains `VALIDATION_0.2.5.md` with checks and measurements.
