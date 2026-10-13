# Changelog

## 0.2.4 - 2026-10-13

Changes since 0.2.3:

- Instance camera-facing atom outlines, coloured half-bonds, polyhedron fill
  triangles, physical edges and engraving bonds using shared meshes.
- Keep geometry resident during camera changes; update edge sorting through
  compact index lists and split large texture buffers at driver limits.
- Add atomic indexed external updates for atom positions/visibility and
  polyhedron replacement, with effective-state getters and reset.
- Patch affected atom records and existing adjacent bonds; give modified
  polyhedra independent buffers and refresh dependent section/hatch caches.
- Match atom/face intersection curves to the face colour in coloured mode,
  including polyhedron colour overrides; retain black ink in engraving mode.
- Refresh contact colours after display-mode, style or polyhedron changes.
- Preserve input Scene, camera, existing API/models, line widths, intact-polygon
  hatching and OpenGL 3.3 support. Add documentation and an external-control example.

## 0.2.3.03 - 2026-10-13 (local experiment)

- Add atomic indexed `update_scene` packets, atom coordinate/visibility setters,
  polyhedron updates, reset and effective-state getters.
- Keep original Scene data and camera unchanged; preserve host style restrictions.
- Patch only changed atom/outline rows and existing adjacent bond records.
- Give replaced polyhedra independent buffers; refresh only dependent sections
  and hatching, preserving intact faces, transparency ordering and line styles.
- Retain clipped stable slots for hidden atoms and degenerate bonds.
- Coalesce GPU updates until paint and release dynamic resources with scene/context.
- Include an external-control example and preserve existing API signatures.
- Python packaging normalizes `0.2.3.03` to `0.2.3.3`.

## 0.2.3.02 - 2026-10-12 (local experiment)

- Render coloured half-bonds with one shared indexed cylinder.
- Render polyhedron fill triangles, physical edges and engraving bonds
  with shared primitives and compact instance records.
- Preserve face/edge transparency ordering, physical shared-edge deduplication,
  lighting, line widths, antialiasing and depth offsets on OpenGL 3.3.
- Batch depth-sorted edges with a small integer order list while keeping
  coordinates and colours resident.
- Split primitive instance storage at the driver's texture-buffer limit.
- Keep hatching and atom/face intersections unchanged; retain intact polygon
  faces for hatching and preserve the public API.
- Extend GL resource cleanup and reinitialization for the new passes.
- Python packaging normalizes `0.2.3.02` to `0.2.3.2`.

## 0.2.3.01 - 2026-10-08 (local experiment)

- Render camera-facing atom silhouettes with a shared instanced ring.
- Keep outline instance buffers resident during rotation, zoom and resize;
  update camera transforms and cosmetic radius padding in the vertex shader.
- Preserve solid/dashed outlines, depth testing, antialiasing and the existing
  order relative to opaque and transparent scene geometry.
- Clean up and recreate all new GL resources with the viewer context.
- Python distribution metadata normalizes this version to `0.2.3.1`.

## 0.2.3 - 2026-10-06

### Performance

- Render atoms with one indexed unit-sphere mesh and one compact instance
  record per visible atom using `glDrawElementsInstanced`.
- Replace repeated per-vertex centre, radius, colour and occupancy data with
  144-byte instance records; keep the shared mesh and index buffers resident
  during atom scale, colour, occupancy visibility and scene changes.
- Remove individual CPU sphere tessellation and the per-atom mesh cache.

### Compatibility

- Preserve the public scene models, viewer aliases, display options, style
  overrides, orientation synchronization and interaction signals.
- Keep OpenGL 3.3 core and GLSL 330 requirements, with 13 attribute locations
  and no new dependencies or native extension.
- Preserve the original sphere triangles and normals, mixed/partial occupancy
  materials, hidden occupancy sectors, external atoms, colour and engraving
  modes, and the existing outline/intersection/hatching passes.
- Release and recreate mesh, index and instance buffers with viewer context
  cleanup/reinitialization.

## 0.2.2 - 2026-10-01

### Fixed

- Individual polyhedron visibility changes now filter resident face, edge and
  atom/face-intersection ranges at draw time, instead of rebuilding their
  geometry and re-uploading the complete buffers.
- Apply the same visibility selection to opaque and transparent faces, edges,
  hatching and intersection contours, including multiple polyhedra sharing a
  site key.

### Performance

- Cache engraving hatching separately for each polyhedron. Reuse GPU buffers
  when the view and effective division count are unchanged; rebuild only the
  affected polyhedra when visible depth extrema alter the density.
- Preserve the original hatching algorithms and density/GRIP normalization
  over visible polyhedra. Keep at most four CPU density variants per polyhedron
  for the current view; discard them when the view or scene changes.
- Vectorize finite-face clipping of sphere/face section segments while keeping
  the original 160-sample circles and boundary tolerance.
- Defer intersection and hatching preparation until their layers are visible.
- Release per-polyhedron hatching GPU resources on scene replacement/clearing
  and context cleanup; recreate them when needed.

### Compatibility

- Keep all public scene models, display options, style overrides, viewer aliases
  and orientation/interaction signals unchanged. Hosts using the existing
  0.2.x API do not need integration changes.

## 0.2.1 - 2026-09-15

### Fixed

- Configure one global OpenGL 3.3 surface format and enable shared OpenGL
  contexts before `QApplication` construction. This prevents stale top-level
  composition after moving or maximising a host window on mixed-DPI screens.
- Release GPU resources when Qt destroys a viewer context and rebuild every
  buffer and shader when a new context is initialised.
- Reduce stair-stepping with 8-sample multisampling and pixel-coverage
  antialiasing for polyhedron edges, hatching, bonds and atom contours.
- Double the period of the dashed unit-cell box while preserving the 2:1
  dash-to-gap ratio.

## 0.2.0 - 2026-09-14

### Changed

- Replaced the legacy QPainter viewer backend with an OpenGL 3.3 renderer using
  a real depth buffer while preserving the existing `UnitCellViewer` and
  `CrystalCanvas` public API.
- `UnitCellViewer`, `CrystalCanvas` and `OpenGLUnitCellViewer` now resolve to
  the same OpenGL-backed widget.
- Render coloured atoms as real 3D spheres and coloured bonds as 3D cylinders;
  engraving bonds remain thin grayscale screen-space lines.
- Keep occupancy sectors camera-facing; unoccupied or hidden occupancy is shown
  in white. Engraving atoms use flat monochrome fills, with dashed outlines for
  partial occupancy.
- Render unit-cell edges with the original dashed style in both colour and
  engraving modes.
- Render polyhedron edges from shared edge geometry and offset them along the
  exterior dihedral bisector so outlines remain visible at obtuse face angles.
- Reduced coloured bond thickness by a factor of three for realistic crystal
  structures.

### Added

- Depth-buffered occlusion between atoms, bonds and polyhedron faces.
- Transparent polyhedron faces with stable screen-space outlines independent of
  driver-specific wide-line support.
- Depth-tested atom outlines and atom/polyhedron-face intersection contours in
  both colour and engraving modes.
- Dynamic engraving hatching that updates with rotation while reusing the
  original 0.1.0 hatching algorithms.
- An explicit `OpenGLUnitCellViewer` name alongside the compatibility aliases.

### Fixed

- Corrected local overlap between spheres, bonds and polyhedron faces that could
  not be represented reliably by whole-primitive painter ordering.
- Fixed hatching disappearing on newly visible faces during rotation.
- Fixed z-fighting between hatching and its supporting face.
- Fixed polyhedron edge outlines becoming partially hidden, including at obtuse
  dihedral angles.
- Fixed atom outlines and atom/face intersection contours appearing on top of
  transparent faces instead of underneath them.
- Avoided unsupported `glLineWidth()` values by expanding important lines into
  screen-space quads in a geometry shader.

### Performance

- Cached sphere, cylinder and face tessellation instead of rebuilding geometry
  during interactive rotation.
- Uploaded static geometry to GPU buffers and moved orientation changes to
  shader uniforms.
- Split render caches so display switches do not rebuild the complete scene.
- Made polyhedron visibility a draw-only toggle instead of an expensive geometry
  rebuild.
- Added broad-phase filtering for atom/face intersection calculations.

### Compatibility

- Preserved scene models, display options, style overrides, orientation events,
  interaction signals and existing construction/import paths.
- Moved the original hatching and screen-drag helper algorithms into shared
  helpers and reused them directly from the OpenGL renderer.

## 0.1.0 - 2026-09-11

- Added immutable input models for caller-supplied atoms, bonds, unit cells,
  basis vectors and ready polyhedra.
- Added native PySide6 rendering with depth ordering, colour and monochrome
  atoms, mixed occupancies, bonds, cell edges and basis arrows.
- Added polyhedron hatching, including intact quadrilateral faces with
  `2x + 1` strokes and direction transport through real shared edges.
- Added left-button rotation with complete orientation-matrix events,
  right-button panning, wheel zoom and a switch for disabling rotation.
- Added global display options and copied per-site style overrides.
- Set polyhedra to hidden by default.
