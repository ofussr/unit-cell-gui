"""Qt/OpenGL setup shared by every unit-cell viewer instance.

Qt must know the common OpenGL format and context-sharing policy before the
``QApplication`` is constructed. This is especially important when a host
window contains more than one ``QOpenGLWidget`` or moves between screens with
different device-pixel ratios.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QSurfaceFormat


def preferred_surface_format() -> QSurfaceFormat:
    """Return the OpenGL format required by the renderer."""

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(8)
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    return fmt


def configure_qt_opengl() -> bool:
    """Configure Qt for compatible, shared ``QOpenGLWidget`` contexts.

    The function is idempotent. It returns ``True`` when it ran before a
    ``QCoreApplication`` was created, which is the fully supported setup. A
    host may call it explicitly during start-up; importing either public
    viewer class also calls it automatically.
    """

    before_application = QCoreApplication.instance() is None
    if before_application:
        QCoreApplication.setAttribute(
            Qt.ApplicationAttribute.AA_ShareOpenGLContexts,
            True,
        )
    QSurfaceFormat.setDefaultFormat(preferred_surface_format())
    return before_application


__all__ = ["configure_qt_opengl", "preferred_surface_format"]
