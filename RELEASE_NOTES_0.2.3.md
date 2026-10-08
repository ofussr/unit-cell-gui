# unit-cell-gui 0.2.3

This release reduces atom geometry preparation and memory use by sharing one
indexed sphere mesh across every visible atom. Existing host applications can
keep their current integration code.

## Atom rendering

- Draw the complete atom layer with `glDrawElementsInstanced`.
- Store each atom's centre, radius, eight RGB colours and cumulative occupancy
  thresholds once in a 144-byte instance record.
- Keep one base sphere and its index buffer per viewer, totaling 18,792 bytes.
  Retain the original 16-by-24 tessellation and all 720 triangles per sphere;
  exact indexing preserves the original triangle order and normals.
- Remove CPU generation and caching of separate sphere meshes for every atom.
- Reuse the base sphere during scene/style changes and recreate all new GPU
  resources after OpenGL context cleanup.

For 1,000 atoms, the main atom geometry buffers shrink from 328,320,000 bytes
(328.32 MB) to 162,792 bytes (approximately 0.163 MB), including the shared mesh
and indices. This excludes bonds, atom outlines, intersections, polyhedra and
Qt/driver overhead. Large scenes can still spend time on those other layers.
The sphere triangle count has not changed; this is not a frame-rate guarantee.

## Compatibility

- Public API, scene models, viewer aliases, display/style options and signals
  remain unchanged from 0.2.2.
- OpenGL 3.3 core and GLSL 330 remain sufficient; 13 vertex attributes are used,
  below the guaranteed minimum of 16. No new dependencies or C++ extension.
- Preserve camera-facing mixed and partial occupancy sectors, hidden sectors,
  colour overrides, external-atom filtering, atom scaling and engraving fills.
- Keep bonds, contours, intersections, polyhedra and hatching on their existing
  rendering paths.

## Validation

- 48 tests passed with Mesa software OpenGL forced to OpenGL 3.3 / GLSL 330,
  including actual indexed instanced drawing, attribute divisors, index binding,
  buffer sizes, cleanup and reinitialization. All 48 also passed using the
  installed wheel, with Python sources compiled successfully.
- 20 framebuffer comparisons against commit `81f76a1` (0.2.2): 19 were
  pixel-identical. One differed at one pixel by one channel step out of 255,
  consistent with float32 transformation rounding. Cases cover colour/engraving,
  eight and more than eight components, partial/hidden occupancy, external
  atoms, rotation, scaling, transparent/opaque polyhedra and quadrilateral
  hatching.
- Installed-wheel smoke check: two viewers embedded in one host window,
  synchronized orientations and continued rendering after one viewer closes.
- In one local synthetic scene containing 1,000 single-component atoms, first
  CPU atom preparation decreased from 7.063 s to 0.00984 s; repeated preparation
  decreased from 0.410 s to 0.00842 s. These single-run measurements exclude
  GPU rendering and other scene layers; they are not hardware timing guarantees.
