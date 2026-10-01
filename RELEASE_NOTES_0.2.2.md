# unit-cell-gui 0.2.2

This maintenance release fixes pauses when hiding or showing individual
polyhedra through `StyleOverrides.polyhedron_site_visibility`.

## Faster individual visibility changes

- Keep polyhedron faces, edges and atom/face intersection contours in GPU
  buffers and select visible site ranges during drawing.
- Cache engraving hatching per polyhedron and reuse unchanged GPU buffers.
  When hiding a depth-extreme polyhedron changes other polyhedra's density,
  update only the affected hatch geometry.
- Preserve the original visible-depth normalization, density and GRIP controls,
  shared-edge hatch directions and intact quadrilateral faces.

## Initial preparation and resource handling

- Vectorize sphere/face section clipping without reducing the original
  160-sample resolution or changing boundary tolerance.
- Prepare intersection contours and hatching only when their layers are needed.
- Bound CPU hatching variants for the current view and release per-polyhedron
  GPU buffers when a scene is replaced, cleared or its context is cleaned up.

## Compatibility

The public API is unchanged: `UnitCellViewer`, `CrystalCanvas`,
`OpenGLUnitCellViewer`, scene models, display options, style overrides,
orientation synchronization and interaction signals remain compatible with
0.2.1. No host-application changes are required for this visibility fix.

## Validation

- 41 tests passed with a real OpenGL context, including visibility, clipping,
  hatching, resource cleanup and context-rebuild checks.
- 15 framebuffer comparisons against 0.2.1 were pixel-identical, covering
  colour and engraving views, transparent and opaque polyhedra, individual and
  global visibility, rotation, atom scaling, external atoms and quadrilateral
  hatching.
- In one synthetic nine-octahedron scene on Mesa software OpenGL, median
  individual-toggle time decreased from approximately 544 ms to 15 ms.
  This is a local benchmark, not a timing guarantee for other structures or
  hardware.
