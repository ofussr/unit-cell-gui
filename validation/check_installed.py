"""Check the installed package and external-host example on a real GL display."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import shiboken6
import unit_cell_gui
from PySide6.QtWidgets import QApplication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-root', type=Path,
                        help='Require imports to come from this directory (e.g. a pip --target install)')
    parser.add_argument('--expected-version', default='0.2.5')
    parser.add_argument('--output', type=Path, help='Optional JSON result file')
    args = parser.parse_args()
    package = Path(unit_cell_gui.__file__).resolve()
    if args.install_root:
        assert package.is_relative_to(args.install_root.resolve()), str(package)
    assert unit_cell_gui.__version__ == args.expected_version
    example_path = Path(__file__).resolve().parents[1] / 'examples' / 'partial_updates.py'
    spec = importlib.util.spec_from_file_location('example_host', example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    app = QApplication.instance() or QApplication([])
    host = example.Host()
    try:
        host.show()
        app.processEvents()
        assert not host.viewer.grabFramebuffer().isNull()
        host.opacity.setValue(35)
        host.x.setValue(20)
        host.poly.setValue(110)
        host.viewer.grabFramebuffer()
        for quality in (2, 1):
            host.viewer.set_display_options(transparency_supersampling=quality)
            assert not host.viewer.grabFramebuffer().isNull()
            assert host.viewer.opengl_error is None, host.viewer.opengl_error
            assert host.viewer.get_polyhedron_opacities()[0] == .35
            np.testing.assert_array_equal(host.viewer.get_atom_positions()[1], [1.2, 0, 0])
            assert not hasattr(host.viewer, '_depth_peeler')
        for value in (100, 0):
            host.opacity.setValue(value)
            host.viewer.grabFramebuffer()
            assert host.viewer.get_polyhedron_opacities()[0] == value/100
        host.reset_changes()
        host.viewer.grabFramebuffer()
        assert host.viewer.get_polyhedron_opacities()[0] == .64
        np.testing.assert_array_equal(host.viewer.get_atom_positions(), host.original.atom_centers)
        assert host.viewer.opengl_error is None, host.viewer.opengl_error
    finally:
        host.close()
        shiboken6.delete(host)
        app.processEvents()
    result = {'version': unit_cell_gui.__version__, 'package': str(package),
              'external_host_updates_and_reset': 'passed', 'legacy_transparency_setting_accepted': 'passed'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
