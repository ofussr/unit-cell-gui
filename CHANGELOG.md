# Changelog

## 0.2.0 - 2026-09-14

### Changed

- Replaced the legacy QPainter rendering backend with an OpenGL 3.3 renderer
  while preserving the existing `UnitCellViewer` and `CrystalCanvas` public API.
- `UnitCellViewer`, `CrystalCanvas` and `OpenGLUnitCellViewer` now resolve to
  the same OpenGL-backed widget.
- Reimplemented coloured atoms as true 3D spheres and coloured bonds as 3D
  cylinders with depth-buffered rendering.
- Ported the existing polyhedron rendering and black-and-white hatching from
  0.1.0 to the OpenGL backend while preserving their established behaviour.
- Preserved camera-facing occupancy sectors, mixed occupancies, monochrome
  rendering, atom outlines, polyhedron transparency and dashed unit-cell edges.
- Reimplemented polyhedron edge rendering using shared edge geometry and
  exterior dihedral offsets for improved visibility.
- Reduced coloured bond thickness for realistic crystal structures.

### Added

- Real per-fragment depth-buffered occlusion between atoms, bonds and
  polyhedron faces.
- Atom/polyhedron-face intersection contours in colour and engraving modes.
- An explicit `OpenGLUnitCellViewer` public name alongside the compatibility
  aliases.

### Performance

- Cached sphere, cylinder and polyhedron tessellation instead of rebuilding
  geometry during interaction.
- Uploaded static geometry to GPU buffers and moved orientation changes to
  shader uniforms.
- Split rendering caches so display-option changes do not rebuild the complete
  scene.
- Made polyhedron visibility a draw-only toggle.
- Added broad-phase filtering for atom/polyhedron-face intersection
  calculations.

### Compatibility

- Preserved the existing scene models, display options, style overrides,
  orientation events, interaction signals and construction/import paths.
- Reused the original 0.1.0 hatching algorithms and related geometric helpers
  in the OpenGL renderer.

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
