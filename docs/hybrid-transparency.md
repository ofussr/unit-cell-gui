# Weighted transparency (0.2.5)

The default and engraving renderers are unchanged. `DisplayOptions` adds two
trailing booleans `weighted_blended_oit_when_rotate=False` and
`weighted_blended_oit=False`. The existing setter
accepts this field. No external host changes or persistence mechanism are needed.

## Permanent mode

The second Shift + right-button checkbox, **Weighted Blended OIT always**, maps
to `viewer.set_display_options(weighted_blended_oit=True)`. It uses the same
renderer even at rest and after mouse release, with no idle-to-exact timer.
Both checkboxes are independent. Permanent mode takes priority when both are
true; disabling it retains the rotation-only preference. With both false the
sorted renderer is always used. Both are ignored by engraving and use the same
nonfatal fallback/resources. Switching modes does not rebuild geometry or emit
orientation signals. Permanent mode draws when a frame is requested; it does
not introduce a continuous animation/redraw loop.

## Interaction

Shift + right-button press opens a nonblocking QMenu before normal pan handling.
It does not establish a drag. Ordinary right-button dragging pans as before.
The checkbox reflects the individual widget's current display option.

With permanent mode off, a valid left-button drag starts an interaction, but fast rendering starts only
when the orientation actually changes. Internal mouse calls mark their source
while using the existing setter. The fast path remains active while the button
is held; release requests the exact last camera orientation. Ungrab, hide,
focus/window deactivation and another drag terminate an interrupted interaction.

External orientation changes start/restart a 150 ms precise single-shot QTimer.
Equality is evaluated at float32 precision, matching the orientation uniform;
validation and the stored float64 orientation retain their existing behavior.
An unchanged effective orientation does not prolong the series. The timer does
no GL work: it clears the external-motion flag and calls update. A new movement
before an exact paint cancels that pending quality change. No queue of past
orientations is processed. Disabling rotation-only mode, engraving, scene clear and
cleanup stop motion/timers. `rotation_enabled` still gates mouse rotation;
external `set_orientation` can rotate with mouse rotation disabled.

`emit=False`, `orientation_changed`, and normal-release `interaction_finished`
keep their meanings. Changing quality emits no orientation or interaction signal.
External movement does not synthesize an interaction_finished signal.

## Depth and MSAA prototype

`validation/prototype_weighted_oit.py` validates the chosen scheme before scene
integration: a native opaque triangle and its WBOIT-base image are pixel-identical
with zero and four requested MSAA samples; an occluded transparent layer is
rejected. The implementation uses only OpenGL 3.3 functionality, with no
independent blend extension and no depth blit between incompatible Qt targets.

The component owns three colour textures, three FBOs, one depth renderbuffer,
a fullscreen VAO and four programs (surface, edge, single/MS composition).
Texture-buffer geometry and its VAOs remain owned by the widget. The GL program
compiler helper is shared, but each resulting handle has exactly one owner.

The target size is the actual physical Qt viewport, and samples are queried
while `defaultFramebufferObject()` is bound. All owned targets have that size
and actual sample count. Their attachments are:

| Target | Colour | Shared depth | Bytes/pixel/sample |
| --- | --- | --- | --- |
| Opaque scene | RGBA8 | DEPTH_COMPONENT24 (accounted as 4 bytes) | 8 |
| Weighted accumulation | RGBA16F | same renderbuffer | 8 |
| Revealage | R32F | same renderbuffer | 4 |

Framebuffer completeness and limits are checked. A mismatching promoted depth
sample count is rejected. No depth or colour handle is borrowed from Qt.
Opaque geometry is rendered directly into the owned scene target during a fast
frame. The composition shader samples all three textures at each sample index,
combines transparency with that sample's opaque colour, then averages final
colours and writes the result to Qt's actual framebuffer. Averaging accumulation
and revealage *before* compositing would give different edge results, so it is
not used. In the single-sample variant, texelFetch uses mip level zero.

## Layer semantics and equations

Atoms, coloured bonds and opacity-one polyhedra write depth. Normal atom
silhouettes and contacts retain the exact renderer's depth-tested ink-base
placement, with no depth writes; transparent surfaces attenuate that base.
Opacity-one polyhedron edges are blended over the opaque base with coverage and
no depth writes (their face already supplied depth). They are not automatically
sent to weighted passes. Opacity-zero faces/edges are omitted. Only fractional
faces and their physical edges enter the unordered passes. Edge output alpha
includes the original geometry shader's feather coverage.

For each accepted fragment with colour c and effective alpha a:

- accumulation uses additive blending: `(c*a, a) * w`;
- revealage starts at one and multiplies by `(1-a)` in the second pass;
- weighted colour is `accum.rgb / max(accum.a, 1e-8)`;
- each final sample is `weighted_colour*(1-reveal) + opaque_colour*reveal`.

The orthographic projection uses reversed window depth (nearer is larger), so
`w = (0.1 + 0.9*z^3) * min(1, 1024/U)`, with z clamped to [0,1] and U at least
one. U is the total transparent triangle count plus twice the expanded edge
count, a conservative bound on contributions to a sample. RGB/alpha and the
unscaled depth weight are bounded by one, so accumulation before rounding is bounded by 1024,
well below half-float overflow. Common scaling cancels during colour recovery.
Revealage uses float32 so many small opacity values do not disappear due to
half-float rounding near one. This guards overflow, not arbitrarily accurate sums: half-float rounding can
lose small contributions in very deep scenes, and revealage can underflow toward
opaque. WBOIT is an approximation even with infinite precision. It can change
hues or edge emphasis when the exact frame replaces it.

The method follows the accumulation/revealage structure of
[McGuire and Bavoil, JCGT 2013](https://jcgt.org/published/0002/02/09/).
The bounded reversed orthographic depth weight above is this implementation's
choice, not the paper's perspective depth formula. Relevant API requirements:
[OpenGL 3.3 specification](https://registry.khronos.org/OpenGL/specs/gl/glspec33.core.pdf)
and [QOpenGLWidget framebuffer lifecycle](https://doc.qt.io/qt-6/qopenglwidget.html).

## Ownership, fallback and performance

Resources are lazy, reused after release and rebuilt only for size/sample
changes. Target storage is approximately `20*width*height*samples` bytes,
excluding Qt buffers, mesh data and driver padding. A 256 MiB per-widget limit
refuses larger targets; it does not reduce MSAA or silently truncate objects.
Default-off operation allocates nothing for WBOIT. Scene clear frees attachments
on the next GL paint, cleanup frees all component objects, and context recreation
starts with a fresh component. Close/show restores through the existing lifecycle.

Partial initialization/allocation failures delete registered handles. A failed
optional frame restores state and clears Qt's target before exact fallback.
`weighted_oit_error` stores its separate nonfatal diagnosis; failed WBOIT is not
retried each frame. The menu displays unavailability/reason, and a new context
or close/show permits retry. The normal OpenGL diagnostic is not overwritten.
Framebuffer/read target, viewport, program/VAO, touched texture units/targets,
renderbuffer, depth/blend settings, write masks and changed enables are restored.

Fast frames bypass preparation of the exact order and the dependency graph.
They reuse resident instance records, coalesce storage-contiguous ranges and
make two unordered geometry passes plus composition. Common geometry updates
are still applied when the host actually changes a scene. With permanent mode off, end-of-motion sorting
runs once for the last orientation, unless an existing exact cache already fits.

Two extra transparent passes, per-sample composition and memory bandwidth are
real costs. On small scenes they can outweigh saved CPU sorting. Opt in based
on measurements on the target GPU. The supplied benchmark separates CPU submit,
GPU elapsed queries, frame intervals, draw/sort/upload counts, final-presentation
latency and attachment bytes without readback in the rotation loop.
