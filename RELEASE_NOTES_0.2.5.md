# unit-cell-gui 0.2.5

Changes relative to 0.2.4. This release corrects transparency ordering between
overlapping polyhedra and adds external per-polyhedron opacity updates without
rebuilding scene geometry. Existing host calls remain supported.

## Correct transparent overlap

Transparent faces and physical edges are now composited together in per-pixel
depth order using front-to-back depth peeling. Rear edges are attenuated by
nearer faces instead of appearing as bright overlays. Intersecting polygons
can have different orders in different pixels; sorting by face centres is no
longer used for the transparent pass.

Peeling stops when an occlusion query finds no remaining fragments, without
a fixed layer cutoff. Opaque atoms, bonds and faces provide the base depth.
Resident instanced meshes are reused for every pass. Two-times supersampling
and conservative scissor regions support antialiasing and limit accumulation
work. Sized targets are created lazily, reused across frames, and released
on layer/scene/context teardown.

## External opacity updates

```python
viewer.set_polyhedron_opacities({3: 0.35, 7: 1.0})
viewer.update_scene(
    atom_positions={17: (x, y, z)},
    polyhedron_opacities={3: 0.5},
)
viewer.set_polyhedron_opacities({3: None})  # restore this object's default
opacities = viewer.get_polyhedron_opacities()  # independent effective-state copy
```

Indices refer to the original Scene.polyhedra. Values must be finite real
numbers from 0 to 1: 0 hides the polyhedron and 1 makes it opaque. `None`
removes an override. Invalid packets reject every included change atomically.
Calls run on the GUI thread, coalesce until paint and preserve Scene/camera.

- One float per polyhedron is stored in a separate GPU material table.
  Changing one opacity uploads four bytes; adjacent changes are batched.
- Moving between transparent and opaque passes reuses resident geometry.
  No mesh generation, vertex-buffer replacement or Scene resubmission is needed.
- Fill and physical-edge opacity change together; explicit overrides also
  fade contact curves. Zero suppresses faces, edges, hatches and contact curves.
- Explicit indexed opacity overrides the occupancy-derived default and the
  site's opaque styling in coloured mode. Host visibility restrictions remain.
- In engraving mode, positive values retain opaque fills and black ink; zero
  hides the polyhedron. Requested fractional values remain stored for colour mode.
- `reset_scene_updates()` clears opacity overrides along with earlier partial
  changes; `set_scene()` starts a fresh state. Overrides survive style changes
  and context recreation. Unset values preserve the existing default appearance.

See [the API contract](docs/partial-updates.md) and
[the updated external-control example](examples/partial_updates.py).

## Compatibility and cost

Public models, viewer aliases, dependencies and existing calls are retained.
The new keyword is optional; no input Scene arrays are mutated. OpenGL 3.3
core / GLSL 330 remains sufficient. Atom tessellation and hatching algorithms
are unchanged; the transparent path has a different antialiasing profile.

Depth peeling requires repeated geometry passes and completion queries.
Many overlapping layers can be substantially more expensive than the old
incorrect blending order. Seven offscreen textures use about 49 MB at
640 × 480 physical pixels with two-times supersampling, excluding driver
overhead; doubled device pixel ratio increases storage fourfold. Opaque-only
views and engraving keep the previous rendering path.

The release archive includes `VALIDATION_0.2.5.md` with source/wheel tests,
analytic overlap checks, material upload sizes, images and measurements.
Windows and hardware GPU drivers are not tested in the build environment;
the preceding local peeling build was reported working by the user with
no noticeable slowdown on their tested scene.
