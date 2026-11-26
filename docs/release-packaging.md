# Release packaging

The package uses Maturin and PyO3 with the CPython stable ABI (3.10+). Wheels
are specific to OS and architecture. A compatible wheel does not require Rust
or a compiler. Source builds need a Rust compiler with edition 2024 support
(Rust 1.85 or later; the exact minimum toolchain has not been tested), Maturin
and a platform linker. The source archive vendors the crates pinned in Cargo.lock.

## Local verification

Install Python dependencies and test tools (`numpy`, `PySide6-Essentials`,
`pytest`, `PyOpenGL`, `maturin`). Run `python build.py --fresh --verify` from
an extracted source archive with a native Qt/OpenGL display. On Windows the
helper discovers installed MSVC x64 tools; on Linux use Xvfb/Mesa if needed.
The local Linux wheel is tagged `linux_x86_64` on an x86-64 host; it is not a
portable PyPI distribution wheel.

## Distribution builds

The build-only GitHub Actions workflow prepares Windows x64, Linux x86-64
(manylinux 2.28), and macOS Intel/Apple Silicon wheels and a source distribution.
It checks metadata and imports the installed wheels. It does not publish or
create releases. The macOS builds and manylinux target must be verified on CI;
import checks do not establish graphical compatibility. macOS needs a desktop
OpenGL context and may have different extension support.

Before publishing, confirm matching package version, native ABI and platform
tags; run rendering tests on supported hosts and check the licenses. Run
`python -m twine check --strict` on the assembled distribution files.
Publication is a separate explicit step after release approval. Retain the
existing version-specific DOI for the previous release; update citation metadata
only once the new release date and DOI are known.
