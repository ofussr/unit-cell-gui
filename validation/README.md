# Reproducing the 0.2.5 checks

Run from the extracted project root with the package dependencies and pytest.
Rendering checks require a working OpenGL 3.3 desktop display. The Qt offscreen
or minimal platform skips GL tests; a headless Linux host needs Xvfb/Mesa.
Test modules default to offscreen unless a platform is explicitly selected.
For real GL checks, set the native platform before the commands below:

```powershell
$env:QT_QPA_PLATFORM = "windows"
```

On a Linux desktop or Xvfb display:

```sh
export QT_QPA_PLATFORM=xcb
```

```sh
python -m pip install -e .
python -m pytest -q tests
python validation/check_installed.py
python validation/render_regression_scenes.py --output /tmp/frames
python validation/benchmark_transparency.py --output /tmp/timings --frames 32
python validation/benchmark_sort.py --output /tmp/cpu-sort.json
python validation/benchmark_updates.py --output /tmp/cpu-updates.json
python validation/benchmark_updates.py --gl --output /tmp/frame-updates.json
python validation/benchmark_dense_sort.py --output /tmp/dense-sort.json
```

Output paths are explicit; use another writable directory on Windows. For a
wheel installed using pip --target, put that directory first on PYTHONPATH
and pass its path to check_installed.py --install-root PATH. Scripts locate
bundled fixtures and the host example relative to themselves.

The benchmark uses nonintersecting convex polyhedra, including twelve cubes
sharing boundaries. It measures rotation plus framebuffer readback, excludes
two warm-up frames and records individual samples, uploads, renderer and
unresolved-sort counts in JSON. Repeat runs on the same renderer and DPI.
The old command-line quality/query switches are accepted only for comparisons
with historical checkouts; they do not enable peeling in this build.

The 46 image-regression scenes also retain older intersecting synthetic cubes.
Those are stress/compatibility observations, not proof of correct ordering
outside the nonintersecting-face contract. The geometric-order and shared-face
tests use supported inputs and independently checked overlap colours.

```sh
python -m build
python -m twine check --strict dist/*
```

The release ZIP includes recorded checks and images. Tests cover the stated
Linux/Mesa environment; Windows/hardware GPU validation remains external.
The current version is 0.2.5.

benchmark_sort.py reports CPU ordering alone, excluding Qt, GPU work and image
readback. It records 30 samples per case by default, p50/p95, projected bounds
candidate counts, Python/NumPy versions and source hashes. It covers shared
boundaries and sparse layouts. To compare a trusted previous implementation:

```sh
python validation/benchmark_sort.py --output /tmp/cpu-sort.json --reference-source /path/to/previous/src/unit_cell_gui/_transparency_sort.py
```

The optional reference is executed as Python code. Every measured case checks
identical order and diagnostic counts against it. Geometry generation and
candidate counting are excluded from timed intervals. This is a CPU benchmark;
benchmark_transparency.py remains an end-to-end rotation/readback measurement,
not a GPU timer. Neither predicts hardware frame rates from software rendering.


Release notes compare with the published GitHub 0.2.4 release. Interim local
0.2.5.x artifacts are not release baselines. If replacing an earlier local build
that also used 0.2.5, force-reinstall the supplied wheel and restart the app.

benchmark_updates.py records cold opacity updates separately from set_scene
validation, cold atom/face contact preparation, and warmed atom setters plus
affected contact rebuilding. CPU timing excludes GL. Its --gl mode records
first-scene frames after context initialization, warmed atom-move frames and
bytes uploaded for one site colour change. Run each package separately with
identical fixtures. Small scenes can incur index/preparation overhead; report
that alongside larger-scene/local-update gains.

benchmark_dense_sort.py measures parallel nonintersecting planes sharing screen
bounds: candidate pairs, graph constraints, graph storage estimate and peak
tracemalloc allocations. Memory sampling runs separately from timings. It is
not a cap or a guarantee of linear memory; prepared resident data is outside
the per-call traced peak.


## Hybrid transparency checks

Use the native platform setup above, then run:

```sh
python validation/prototype_weighted_oit.py
python -m pytest -q tests/test_z_weighted_oit.py
python validation/render_regression_scenes.py --weighted-mode final --output /tmp/final
python validation/render_regression_scenes.py --weighted-mode fast --output /tmp/fast
python validation/render_regression_scenes.py --weighted-mode permanent --output /tmp/permanent
python validation/benchmark_weighted_oit.py --frames 60 --output /tmp/hybrid.json
```

Repeat rendering and tests with `QT_SCALE_FACTOR=2` (PowerShell:
`$env:QT_SCALE_FACTOR = "2"`). Compare default/final images with 0.2.5.2 images.
Rotation and permanent WBOIT images must agree at the same camera/materials.
Both intentionally differ from exact images; final exact images must match the
previous sorted implementation. Permanent mode must keep WBOIT at rest/release
and perform no exact sort; switching both options off restores exact rendering. The
prototype compares the opaque image pixel-for-pixel with native rendering for
zero/four requested samples and checks a hidden transparent layer.

The new benchmark uses Qt frame delivery and `frameSwapped`, never
`grabFramebuffer`, `glReadPixels` or per-frame query-result waiting in the
measured rotation loop. It excludes three warm-up frames (including lazy WBOIT
creation). CPU preparation/submission, draw calls, sort calls, geometry uploads,
frame intervals, mouse-release-to-exact presentation and external-idle-to-exact
presentation are separate metrics. GL_TIME_ELAPSED queries are collected after
rotation; a final glFinish outside the timed loop allows query cleanup. GPU
measurements are absent when the driver exposes no timer counter. Instrumentation
adds CPU overhead equally to both modes; software-renderer results are not
predictions of the target laptop's speed. The release latency observes actual
presentation; QWidget.repaint() call duration alone would not measure it.

On the target laptop, test at its usual window size/DPR and compare enabled and
disabled rotation, small scenes, many overlapping polyhedra and atoms/contacts.
Check colour changes on release, p50/p95, final-frame latency, attachment memory,
resize, move between screens and close/show. A 256 MiB attachment-limit fallback
is deliberate and reported, not silent quality reduction.
