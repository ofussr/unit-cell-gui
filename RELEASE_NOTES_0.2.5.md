# unit-cell-gui 0.2.5

Changes relative to the published [0.2.4 GitHub release](https://github.com/ofussr/unit-cell-gui/releases/tag/0.2.4).
This release adds external opacity control, global transparent face/edge ordering,
optional weighted blended transparency, faster local updates and input/resource fixes.

## Transparency modes

- Transparent faces and physical edges of all visible polyhedra now share one
  back-to-front order based on depth over their projected overlap. Rear edges
  are attenuated by nearer faces. Opaque geometry writes depth first.
- The default sorted renderer draws directly into Qt's multisample framebuffer,
  preserving hardware antialiasing for atoms and bonds. No depth peeling,
  per-layer completion queries or whole-window supersampling is used.
- **Shift + right mouse button** opens two independent checkboxes:
  **Weighted Blended OIT when rotate** and **Weighted Blended OIT always**.
  Both are off by default, apply per widget and are not saved to disk.
- Rotation-only WBOIT uses approximate transparency while the mouse rotates;
  releasing the button requests the exact sorted frame. External orientation
  changes use the same mode and request exact rendering after 150 ms of inactivity.
- Permanent WBOIT also applies at rest and after mouse release. It takes priority
  when both checkboxes are checked. Disabling it restores the retained
  rotation-only preference, or exact rendering when that preference is off.
- Both settings are available through the existing display-options API:

```python
viewer.set_display_options(weighted_blended_oit=True)  # permanent WBOIT
viewer.set_display_options(weighted_blended_oit=False) # leave permanent mode
viewer.set_display_options(weighted_blended_oit_when_rotate=True)
```

WBOIT uses two unordered OpenGL 3.3 passes with shared owned MSAA depth and
per-sample composition. It bypasses exact sorting and reuses resident instance
geometry. Transparent edge alpha includes feather coverage. Fully opaque edges
are handled separately; atom contours and contacts remain depth-tested beneath
the transparent result. Engraving ignores both settings. Ordinary right-button
pan, zoom, orientation signals and `emit=False` keep their existing meanings.

## External polyhedron opacity updates

```python
viewer.set_polyhedron_opacities({3: 0.35, 7: 1.0})
viewer.update_scene(atom_positions={17: (x, y, z)},
                    polyhedron_opacities={3: 0.5})
viewer.set_polyhedron_opacities({3: None})  # restore this object's default
opacities = viewer.get_polyhedron_opacities()
```

Indices refer to the original Scene.polyhedra. Finite real values from 0 to 1
control individual objects: zero hides the polyhedron; one makes it opaque;
None removes its override. Invalid packets reject all included changes atomically.
Opacity changes update a separate GPU material table, normally four bytes per
changed object, without rebuilding meshes or resubmitting the scene. Fill and
physical edges change together; explicit overrides also fade contact curves.

Explicit indexed opacity takes priority over occupancy-derived defaults and
site opaque styling in colour mode; host visibility restrictions still apply.
In engraving, positive values retain opaque fills/black ink and zero hides the
polyhedron. Fractional requests remain stored for the next colour view.
Getters return independent effective-state copies. Overrides survive style and
context changes; set_scene/reset_scene_updates clear them as documented.

## Preparation and local-update performance

- Prepare camera-independent packed geometry, topology, stable material/geometry
  keys and world normals once; batch camera projection and bounds calculations.
- Reject disjoint transparent candidates through a projected bounds hierarchy;
  use bounded paired edge/face clipping batches while retaining depth tie rules.
- Query refittable atom/polyhedron spatial indices for contact curves and
  affected sections. Reuse atom arrays and cached face plane/basis/clipping data.
- First opacity-only updates no longer copy coordinates, build bond adjacency
  or prepare geometry bounds. Geometry-related state is created lazily.
- Host colour/opaque-style changes refresh affected polyhedra instead of the
  whole layer. Cached legend entries use visible components of visible atoms.

## Validation and resource handling

- Reject lossy fractional/string/bool indices before conversion, while accepting
  NumPy integers. Reject non-finite or GPU-float32-unrepresentable input and
  derived geometry, radii and scales before changing state.
- Reject singular and numerically unstable orientations atomically while
  retaining supported invertible nonorthogonal transforms/reflections.
- Normalize extreme finite occupancy weights without overflowing sums/radii.
- Free created shaders/programs on partial compile/link failure. Scene clear
  releases scene-dependent CPU arrays and GPU storage while keeping shared meshes.
- Restore rendering on ordinary close/show even when Qt keeps the context.
  WBOIT has lazy, independently owned resources, size/sample limits, a 256 MiB
  attachment budget, cleanup on partial failure and nonfatal exact fallback.
  Its unavailability and reason are shown in the menu and weighted_oit_error.

## Compatibility and limits

Existing 0.2.4 calls, viewer aliases, model constructor arguments, signal
semantics and dependencies remain supported. New DisplayOptions fields are
trailing and optional. Previously accepted lossy/overflowing/unstable inputs now
raise validation errors. Atom tessellation and intact-face hatching are unchanged;
outline, bond and polyhedron instancing already present in 0.2.4 is retained.

The sorted path expects planar convex faces without physical intersections or
cyclic visibility constraints. Shared boundaries/coincident materials retain
stable ordering. Dense projected overlap can still require quadratic graph memory.
WBOIT is approximate and can change hues/edge emphasis; permanent mode deliberately
keeps that approximation at rest. Small scenes can be slower with WBOIT enabled.

Additional WBOIT attachment storage is approximately 20 bytes × physical width ×
height × samples, excluding Qt, geometry and driver padding. Buffers are reused.
DPR 2 multiplies storage by four. Exceeding driver/budget limits falls back to
sorted rendering without silently reducing MSAA. OpenGL 3.3/GLSL 330 remains
sufficient; native Windows/hardware/mixed-monitor testing remains an external step.

See [partial-update API](docs/partial-updates.md),
[transparency behavior](docs/hybrid-transparency.md) and
[reproducible checks](validation/README.md). This local preparation publishes nothing.
