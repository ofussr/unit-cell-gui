# Changelog

## 0.1.0

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
