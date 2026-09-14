# Changelog

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
