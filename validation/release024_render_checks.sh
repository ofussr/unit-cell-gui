#!/bin/bash
set -e
PYTHONPATH=outline-baseline/src:unit-cell-gui/tests .venv/bin/python render_instancing02.py release024-frames-023
QT_SCALE_FACTOR=2 PYTHONPATH=outline-baseline/src:unit-cell-gui/tests .venv/bin/python render_instancing02.py release024-frames-023-dpi2
PYTHONPATH=unit-cell-gui/src:unit-cell-gui/tests .venv/bin/python render_release024.py release024-frames-new
QT_SCALE_FACTOR=2 PYTHONPATH=unit-cell-gui/src:unit-cell-gui/tests .venv/bin/python render_release024.py release024-frames-new-dpi2
PYTHONPATH=unit-cell-gui/src:unit-cell-gui/tests .venv/bin/python render_release024.py release024-frames-old-contact old-contact-colour
QT_SCALE_FACTOR=2 PYTHONPATH=unit-cell-gui/src:unit-cell-gui/tests .venv/bin/python render_release024.py release024-frames-old-contact-dpi2 old-contact-colour
