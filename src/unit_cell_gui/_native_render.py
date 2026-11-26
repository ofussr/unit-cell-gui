"""Context-local Rust submission, using Qt's OpenGL function resolver."""
from PySide6.QtGui import QOpenGLContext
from ._ucg_native import GlSubmission

_SYMBOLS = ('glUseProgram', 'glBindVertexArray', 'glActiveTexture',
            'glBindTexture', 'glUniform1i', 'glDrawArraysInstanced',
            'glGetError', 'glUniform1f', 'glDrawArrays', 'glDrawElementsInstanced',
            'glVertexAttribIPointer')


def create_submission(context):
    if context is None or QOpenGLContext.currentContext() is not context:
        raise RuntimeError('OpenGL context is not current')
    return GlSubmission([int(context.getProcAddress(name.encode('ascii')) or 0)
                         for name in _SYMBOLS])
