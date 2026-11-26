"""Shared std140 frame block and context-local native state operations."""
from contextlib import contextmanager
from functools import wraps
import re
import numpy as np
from . import _gl as g
from PySide6.QtGui import QOpenGLContext

SLOT = 7  # minimum guaranteed GL 3.3 binding count is 36
BLOCK = '''layout(std140) uniform UcgFrame {
    mat4 u_projection;
    mat3 u_orientation;
    vec4 ucg_light;
    vec4 ucg_viewport;
};
#define u_light_dir ucg_light.xyz
#define u_viewport ucg_viewport.xy
'''
DECL = re.compile(r'uniform\s+(?:mat4\s+u_projection|mat3\s+u_orientation|vec3\s+u_light_dir|vec2\s+u_viewport)\s*;')
SYMBOLS = ('glGetIntegerv','glGetIntegeri_v','glGetInteger64i_v','glGetBooleanv','glIsEnabled',
           'glUseProgram','glUniform1i','glUniform1f','glBindBuffer','glBufferSubData',
           'glBindBufferBase','glBindBufferRange','glActiveTexture','glBindTexture','glBindFramebuffer',
           'glBindVertexArray','glBindRenderbuffer','glViewport','glDepthFunc','glDepthMask','glColorMask',
           'glBlendFuncSeparate','glBlendEquationSeparate','glEnable','glDisable','glGetError')

def shader(source):
    if source is None or not DECL.search(source):return source
    # Insert exactly once at first former declaration, after any #extension.
    i=DECL.search(source).start()
    return source[:i]+BLOCK+DECL.sub('',source[i:])

def create_program(compile_program,*sources):
    program=compile_program(*(shader(s) for s in sources))
    try:
        idx=int(g.glGetUniformBlockIndex(program,'UcgFrame'))
        if idx != 0xffffffff:
            g.glUniformBlockBinding(program,idx,SLOT)
        return program
    except BaseException:
        g.glDeleteProgram(program)
        raise

class FrameParameters:
    def __init__(self,context,resources=None):
        if context is None or QOpenGLContext.currentContext() is not context:
            raise RuntimeError('OpenGL context is not current')
        self.context=context
        self.resources=resources or g
        from ._ucg_native import GlFrame
        self.native=GlFrame([int(context.getProcAddress(s.encode('ascii')) or 0) for s in SYMBOLS])
        self.buffer=0; self.last=None;self.uploads=0
        old=int(g.glGetIntegerv(g.GL_UNIFORM_BUFFER_BINDING))
        try:
            self.buffer=int(self.resources.glGenBuffers(1))
            self.resources.glBindBuffer(g.GL_UNIFORM_BUFFER,self.buffer)
            self.resources.glBufferData(g.GL_UNIFORM_BUFFER,144,None,g.GL_DYNAMIC_DRAW)
        except BaseException:
            self.cleanup();raise
        finally:self.resources.glBindBuffer(g.GL_UNIFORM_BUFFER,old)

    @contextmanager
    def bound(self,viewer):
        if QOpenGLContext.currentContext() is not self.context:
            raise RuntimeError('frame OpenGL context is not current')
        data=np.zeros(36,np.float32)
        data[:16]=viewer._projection_matrix().T.ravel()
        data[16:28].reshape(3,4)[:,:3]=np.asarray(viewer.orientation,np.float32).T
        data[28:31]=(-.35,.45,.82)
        data[32:34]=viewer._framebuffer_size()
        key=data.tobytes(); changed=key!=self.last
        state=self.native.bind_frame(self.buffer,data.tolist() if changed else None)
        if changed:self.last=key;self.uploads+=1
        try:yield
        finally:self.native.restore_frame(state)

    def cleanup(self):
        if self.buffer:self.resources.glDeleteBuffers(1,[self.buffer]);self.buffer=0
        self.last=None

def frame_scope(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        if self._frame_parameters is None:return fn(self,*args,**kwargs)
        with self._frame_parameters.bound(self):return fn(self,*args,**kwargs)
    return wrapped
