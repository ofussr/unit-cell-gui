"""Small, private GL 3.3 binding. Resolve functions against the current Qt context.
No PyOpenGL runtime dependency. Native Rust batching remains the preferred path.
Only functions used by this package are exposed; not a general OpenGL binding.
"""
import ctypes as C
import weakref
import numpy as np
from PySide6.QtGui import QOpenGLContext

GL_ACTIVE_TEXTURE = 0x84e0
GL_ARRAY_BUFFER = 0x8892
GL_BLEND = 0xbe2
GL_BLEND_DST_ALPHA = 0x80ca
GL_BLEND_DST_RGB = 0x80c8
GL_BLEND_EQUATION_ALPHA = 0x883d
GL_BLEND_EQUATION_RGB = 0x8009
GL_BLEND_SRC_ALPHA = 0x80cb
GL_BLEND_SRC_RGB = 0x80c9
GL_COLOR = 0x1800
GL_COLOR_ATTACHMENT0 = 0x8ce0
GL_COLOR_ATTACHMENT1 = 0x8ce1
GL_COLOR_BUFFER_BIT = 0x4000
GL_COLOR_WRITEMASK = 0xc23
GL_COMPILE_STATUS = 0x8b81
GL_CULL_FACE = 0xb44
GL_CURRENT_PROGRAM = 0x8b8d
GL_DEPTH = 0x1801
GL_DEPTH_ATTACHMENT = 0x8d00
GL_DEPTH_BUFFER_BIT = 0x100
GL_DEPTH_COMPONENT24 = 0x81a6
GL_DEPTH_FUNC = 0xb74
GL_DEPTH_TEST = 0xb71
GL_DEPTH_WRITEMASK = 0xb72
GL_DRAW_FRAMEBUFFER = 0x8ca9
GL_DRAW_FRAMEBUFFER_BINDING = 0x8ca6
GL_DYNAMIC_DRAW = 0x88e8
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_EXTENSIONS = 0x1f03
GL_FALSE = 0x0
GL_FLOAT = 0x1406
GL_FRAGMENT_SHADER = 0x8b30
GL_FRAMEBUFFER = 0x8d40
GL_FRAMEBUFFER_COMPLETE = 0x8cd5
GL_FRAMEBUFFER_SRGB = 0x8db9
GL_FUNC_ADD = 0x8006
GL_GEOMETRY_SHADER = 0x8dd9
GL_GREATER = 0x204
GL_INFO_LOG_LENGTH = 0x8b84
GL_INT = 0x1404
GL_INVALID_INDEX = 0xffffffff
GL_LINES = 0x1
GL_LINK_STATUS = 0x8b82
GL_MAX_COLOR_TEXTURE_SAMPLES = 0x910e
GL_MAX_DRAW_BUFFERS = 0x8824
GL_MAX_RENDERBUFFER_SIZE = 0x84e8
GL_MAX_SAMPLES = 0x8d57
GL_MAX_TEXTURE_BUFFER_SIZE = 0x8c2b
GL_MAX_TEXTURE_SIZE = 0xd33
GL_MULTISAMPLE = 0x809d
GL_NEAREST = 0x2600
GL_NO_ERROR = 0x0
GL_NUM_EXTENSIONS = 0x821d
GL_ONE = 0x1
GL_ONE_MINUS_SRC_ALPHA = 0x303
GL_ONE_MINUS_SRC_COLOR = 0x301
GL_R32F = 0x822e
GL_READ_FRAMEBUFFER = 0x8ca8
GL_READ_FRAMEBUFFER_BINDING = 0x8caa
GL_RED = 0x1903
GL_RENDERBUFFER = 0x8d41
GL_RENDERBUFFER_BINDING = 0x8ca7
GL_RENDERBUFFER_SAMPLES = 0x8cab
GL_RGBA = 0x1908
GL_RGBA16F = 0x881a
GL_RGBA32F = 0x8814
GL_RGBA8 = 0x8058
GL_SAMPLES = 0x80a9
GL_SAMPLE_ALPHA_TO_COVERAGE = 0x809e
GL_SCISSOR_TEST = 0xc11
GL_SRC_ALPHA = 0x302
GL_STATIC_DRAW = 0x88e4
GL_TEXTURE0 = 0x84c0
GL_TEXTURE_2D = 0xde1
GL_TEXTURE_2D_MULTISAMPLE = 0x9100
GL_TEXTURE_BINDING_2D = 0x8069
GL_TEXTURE_BINDING_2D_MULTISAMPLE = 0x9104
GL_TEXTURE_BINDING_BUFFER = 0x8c2c
GL_TEXTURE_BUFFER = 0x8c2a
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_MAX_LEVEL = 0x813d
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TRIANGLES = 0x4
GL_TRIANGLE_STRIP = 0x5
GL_TRUE = 0x1
GL_UNIFORM_BUFFER = 0x8a11
GL_UNIFORM_BUFFER_BINDING = 0x8a28
GL_UNSIGNED_INT = 0x1405
GL_VERSION = 0x1f02
GL_VERTEX_ARRAY_BINDING = 0x85b5
GL_VERTEX_SHADER = 0x8b31
GL_VIEWPORT = 0xba2
GL_ZERO = 0x0

_U = C.c_uint32
_I = C.c_int32
_B = C.c_ubyte
_F = C.c_float
_D = C.c_double
_P = C.c_void_p
_S = C.c_ssize_t
_Q = C.c_int64
_STR = C.c_char_p
_FUN = getattr(C, 'WINFUNCTYPE', C.CFUNCTYPE)
_cache = weakref.WeakKeyDictionary()


def _proc(name, result, args):
    context = QOpenGLContext.currentContext()
    if context is None:
        raise RuntimeError(f'{name}: OpenGL context is not current')
    functions = _cache.setdefault(context, {})
    key = (name, result, args)
    if key not in functions:
        address = int(context.getProcAddress(name.encode('ascii')) or 0)
        if address <= 3 or address == C.c_void_p(-1).value:
            raise RuntimeError(f'{name}: OpenGL entry point unavailable')
        functions[key] = _FUN(result, *args)(address)
    return functions[key]


def _call(name, result, args, *values):
    value = _proc(name, result, args)(*values)
    if name != 'glGetError':
        error = _proc('glGetError', _U, ())()
        if error:
            raise RuntimeError(f'{name}: OpenGL error 0x{error:04x}')
    return value


def _simple(name, result, *args):
    def call(*values):
        return _call(name, result, args, *values)
    call.__name__ = name
    globals()[name] = call


for _name in ('glActiveTexture glBindVertexArray glClear glCompileShader glDeleteProgram glDeleteShader glDepthFunc glDisable glDisableVertexAttribArray glDrawBuffer glEnable glEnableVertexAttribArray glLinkProgram glReadBuffer glUseProgram glBlendEquation'.split()):
    _simple(_name, None, _U)
for _name in ('glAttachShader glBindBuffer glBindFramebuffer glBindRenderbuffer glBindTexture glBlendEquationSeparate glBlendFunc glVertexAttribDivisor'.split()):
    _simple(_name, None, _U, _U)
for _name in ('glTexBuffer glUniformBlockBinding'.split()):
    _simple(_name, None, _U, _U, _U)
for _name in ('glBlendFuncSeparate glFramebufferRenderbuffer'.split()):
    _simple(_name, None, _U, _U, _U, _U)
_simple('glCreateShader', _U, _U)
_simple('glCreateProgram', _U)
_simple('glGetError', _U)
_simple('glIsEnabled', _B, _U)
_simple('glCheckFramebufferStatus', _U, _U)
_simple('glDepthMask', None, _B)
_simple('glClearColor', None, _F, _F, _F, _F)
_simple('glClearDepth', None, _D)
_simple('glColorMask', None, _B, _B, _B, _B)
_simple('glViewport', None, _I, _I, _I, _I)
_simple('glUniform1i', None, _I, _I)
_simple('glUniform1f', None, _I, _F)
_simple('glUniform2f', None, _I, _F, _F)
_simple('glUniform3f', None, _I, _F, _F, _F)
_simple('glDrawArrays', None, _U, _I, _I)
_simple('glDrawArraysInstanced', None, _U, _I, _I, _I)
_simple('glDrawElementsInstanced', None, _U, _I, _U, _P, _I)
_simple('glVertexAttribPointer', None, _U, _I, _U, _B, _I, _P)
_simple('glVertexAttribIPointer', None, _U, _I, _U, _I, _P)
_simple('glTexParameteri', None, _U, _U, _I)
_simple('glTexImage2DMultisample', None, _U, _I, _U, _I, _I, _B)
_simple('glRenderbufferStorage', None, _U, _U, _I, _I)
_simple('glRenderbufferStorageMultisample', None, _U, _I, _U, _I, _I)
_simple('glFramebufferTexture2D', None, _U, _U, _U, _U, _I)
_simple('glBindBufferBase', None, _U, _U, _U)
_simple('glBindBufferRange', None, _U, _U, _U, _S, _S)
_simple('glBlendFuncSeparatei', None, _U, _U, _U, _U, _U)
_simple('glBlendEquationSeparatei', None, _U, _U, _U)
_simple('glBlendFuncSeparateiARB', None, _U, _U, _U, _U, _U)
_simple('glBlendEquationSeparateiARB', None, _U, _U, _U)
_simple('glEnablei', None, _U, _U)
_simple('glDisablei', None, _U, _U)
_simple('glIsEnabledi', _B, _U, _U)


def _gen(name, count):
    out = (_U * count)()
    _call(name, None, (_I, _P), count, out)
    return int(out[0]) if count == 1 else np.asarray(out).copy()


def _delete(name, count, ids):
    ids = tuple(map(int, ids))
    if count < 0 or len(ids) != count:
        raise ValueError("resource count mismatch")
    values = (_U * count)(*ids)
    _call(name, None, (_I, _P), count, values)


for _kind in ('Buffers', 'VertexArrays', 'Textures', 'Framebuffers', 'Renderbuffers'):
    globals()['glGen'+_kind] = lambda n, k=_kind: _gen('glGen'+k,n)
    globals()['glDelete'+_kind] = lambda n, ids, k=_kind: _delete('glDelete'+k,n,ids)


def _query(name, ty, prefix, values, count=1):
    out = (ty * count)()
    _call(name, None, prefix+(_P,), *values, out)
    return int(out[0]) if count == 1 else np.asarray(out).copy()


def glGetIntegerv(pname):
    return _query('glGetIntegerv', _I, (_U,), (pname,), 4 if pname == GL_VIEWPORT else 1)


def glGetBooleanv(pname):
    return _query('glGetBooleanv', _B, (_U,), (pname,), 4 if pname == GL_COLOR_WRITEMASK else 1)


def glGetIntegeri_v(pname, index):
    return _query('glGetIntegeri_v', _I, (_U,_U), (pname,index))


def glGetInteger64i_v(pname, index):
    return _query('glGetInteger64i_v', _Q, (_U,_U), (pname,index))


for _name in ('glGetShaderiv','glGetProgramiv','glGetRenderbufferParameteriv'):
    globals()[_name] = lambda a,b,n=_name: _query(n,_I,(_U,_U),(a,b))


def _log(kind, obj):
    size = globals()['glGet'+kind+'iv'](obj, GL_INFO_LOG_LENGTH)
    out = C.create_string_buffer(max(1,size))
    _call('glGet'+kind+'InfoLog', None, (_U,_I,_P,_P), obj,len(out),None,out)
    return out.value


def glGetShaderInfoLog(obj): return _log('Shader',obj)
def glGetProgramInfoLog(obj): return _log('Program',obj)


def glShaderSource(shader, source):
    encoded = source.encode('utf-8')
    data = (_STR * 1)(encoded)
    _call('glShaderSource', None, (_U,_I,_P,_P), shader,1,data,None)


def glGetUniformLocation(program,name):
    return _call('glGetUniformLocation', _I, (_U,_STR),program,name.encode('ascii'))


def glGetUniformBlockIndex(program,name):
    return _call('glGetUniformBlockIndex', _U, (_U,_STR),program,name.encode('ascii'))


def _matrix(n, location, count, transpose, data):
    a = np.ascontiguousarray(data,dtype=np.float32)
    if a.size != n*n*count: raise ValueError('invalid matrix length')
    _call(f'glUniformMatrix{n}fv', None, (_I,_I,_B,_P),location,count,transpose,a.ctypes.data)


def glUniformMatrix3fv(*args): _matrix(3,*args)
def glUniformMatrix4fv(*args): _matrix(4,*args)


def _buffer(name,target,offset,size,data,usage=None):
    if size < 0 or offset < 0:
        raise ValueError("negative buffer size or offset")
    if data is None and usage is None and size:
        raise ValueError("nonempty buffer update requires data")
    # Use the byte-buffer interface for bytes-like data, not string conversion.
    if isinstance(data, (bytes, bytearray, memoryview)):
        a = np.frombuffer(memoryview(data).tobytes() if isinstance(data, memoryview) and not data.c_contiguous else data, dtype=np.uint8)
    else:
        a = None if data is None else np.ascontiguousarray(data)
    if a is not None and a.nbytes != size: raise ValueError("buffer payload size mismatch")
    ptr = None if a is None else a.ctypes.data
    if usage is not None: _call(name,None,(_U,_S,_P,_U),target,size,ptr,usage)
    else: _call(name,None,(_U,_S,_S,_P),target,offset,size,ptr)


def glBufferData(target,size,data,usage): _buffer('glBufferData',target,0,size,data,usage)
def glBufferSubData(target,offset,size,data): _buffer('glBufferSubData',target,offset,size,data)


def glClearBufferfv(buffer,drawbuffer,data):
    a=np.ascontiguousarray(data,dtype=np.float32)
    if a.size < (4 if buffer==GL_COLOR else 1): raise ValueError('clear values too short')
    _call('glClearBufferfv',None,(_U,_I,_P),buffer,drawbuffer,a.ctypes.data)


def glTexImage2D(target,level,internal,w,h,border,fmt,ty,data):
    if data is not None: raise ValueError('only empty texture allocation is supported')
    _call('glTexImage2D',None,(_U,_I,_I,_I,_I,_I,_U,_U,_P),target,level,internal,w,h,border,fmt,ty,None)


def glDrawBuffers(values):
    a=(_U*len(values))(*values)
    _call('glDrawBuffers',None,(_I,_P),len(values),a)
