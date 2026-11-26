"""Context-local resource API; no global OpenGL monkeypatching."""
from PySide6.QtGui import QOpenGLContext
from . import _gl as g
import os
import sys
import ctypes

_KINDS=('Buffers','VertexArrays','Textures','Framebuffers','Renderbuffers')
_SYMBOLS=tuple('glGen'+k for k in _KINDS)+tuple('glDelete'+k for k in _KINDS)+(
    'glBindBuffer','glBindTexture','glBindVertexArray','glEnableVertexAttribArray',
    'glDisableVertexAttribArray','glVertexAttribPointer','glVertexAttribIPointer',
    'glVertexAttribDivisor','glTexBuffer','glBufferData','glBufferSubData','glGetError',
    'glBindRenderbuffer','glBindFramebuffer','glRenderbufferStorage','glRenderbufferStorageMultisample',
    'glGetRenderbufferParameteriv','glTexImage2D','glTexImage2DMultisample','glTexParameteri',
    'glFramebufferTexture2D','glFramebufferRenderbuffer','glDrawBuffer','glReadBuffer','glCheckFramebufferStatus')

class ResourceGL:
    def __init__(self, context, legacy):
        self.context=context; self.legacy=legacy; self.native=None; self.error=None
        self.borrowed_buffers=False
        self.uploads=0; self.upload_bytes=0; self.patches=0; self.patch_bytes=0
        if context is None or os.environ.get('UNIT_CELL_GUI_RESOURCE_BACKEND','rust').lower()=='python':return
        try:
            if context is None or QOpenGLContext.currentContext() is not context:
                raise RuntimeError('OpenGL context is not current')
            from ._ucg_native import GlResources
            buffer_api=None
            if sys.implementation.name=='cpython' and sys.version_info >= (3,11):
                try:
                    buffer_api=tuple(ctypes.cast(getattr(ctypes.pythonapi,n),ctypes.c_void_p).value
                                     for n in ('PyObject_GetBuffer','PyBuffer_Release'))
                except (AttributeError,TypeError):pass
            self.native=GlResources([int(context.getProcAddress(s.encode('ascii')) or 0) for s in _SYMBOLS],buffer_api)
            self.borrowed_buffers=buffer_api is not None
        except Exception as exc:self.error=str(exc) or type(exc).__name__

    def _check(self):
        if QOpenGLContext.currentContext() is not self.context:
            raise RuntimeError('resource OpenGL context is not current')

    def __getattr__(self,name):
        # Legacy module is resolved at call time (including test spies).
        if self.native is None:return getattr(self.legacy,name,None) or getattr(g,name)
        if name.startswith('glGen') and name[5:] in _KINDS:
            kind=_KINDS.index(name[5:])
            def gen(n):
                self._check()
                if n!=1:raise ValueError('only one resource at a time')
                return self.native.generate(kind)
            return gen
        if name.startswith('glDelete') and name[8:] in _KINDS:
            kind=_KINDS.index(name[8:])
            def delete(n,ids):
                self._check(); ids=list(map(int,ids))
                if n!=len(ids):raise ValueError('resource count mismatch')
                return self.native.delete(kind,ids)
            return delete
        methods={'glBindBuffer':'bind_buffer','glBindTexture':'bind_texture','glBindVertexArray':'bind_vao',
                 'glVertexAttribDivisor':'divisor','glTexBuffer':'tex_buffer'}
        if name in methods:
            def call(*args):self._check();return getattr(self.native,methods[name])(*map(int,args))
            return call
        if name in ('glEnableVertexAttribArray','glDisableVertexAttribArray'):
            def toggle(index):self._check();return self.native.enable_attrib(int(index),name=='glEnableVertexAttribArray')
            return toggle
        if name=='glVertexAttribPointer':
            def attrib(index,size,ty,norm,stride,offset):
                self._check();return self.native.attrib(int(index),int(size),int(ty),bool(norm),int(stride),int(offset.value or 0))
            return attrib
        if name=='glVertexAttribIPointer':
            def attrib_i(index,size,ty,stride,offset):
                self._check();return self.native.attrib_i(int(index),int(size),int(ty),int(stride),int(offset.value or 0))
            return attrib_i
        return getattr(self.legacy,name,None) or getattr(g,name)

    def _payload(self,data):
        view=memoryview(data)
        return view if self.borrowed_buffers and view.c_contiguous else view.tobytes()

    def glBufferData(self,target,size,data,usage):
        if self.native is None:self.legacy.glBufferData(target,size,data,usage)
        else:
            self._check();payload=None if data is None else self._payload(data)
            self.native.buffer_data(int(target),int(size),int(usage),payload)
        self.uploads+=1;self.upload_bytes+=int(size) if data is not None else 0

    def glBufferSubData(self,target,offset,size,data):
        if self.native is None:self.legacy.glBufferSubData(target,offset,size,data)
        else:
            self._check();payload=self._payload(data)
            if memoryview(payload).nbytes!=size:raise ValueError('buffer patch size mismatch')
            self.native.buffer_sub_data(int(target),int(offset),payload)
        self.patches+=1;self.patch_bytes+=int(size)

    def upload_static(self,vao,vbo,vertices):
        self._check();self.native.upload_static(int(vao),int(vbo),self._payload(vertices))
        self.uploads+=1;self.upload_bytes+=vertices.nbytes

    def patch_rows(self,vbo,blocks):
        self._check();payloads=[(int(offset),self._payload(block)) for offset,block in blocks]
        self.native.patch_rows(int(vbo),payloads)
        self.patches+=len(payloads);self.patch_bytes+=sum(memoryview(b).nbytes for _,b in payloads)
