"""Optional GL 3.3 weighted transparency. All GL objects here are owned here.

Three same-sized FBOs share one owned depth renderbuffer. The opaque colour,
accumulation and revealage textures have the *actual* Qt sample count. The
final shader composites each sample before averaging, without a depth blit.
Independent blending optionally combines the two transparent geometry passes.
"""
from __future__ import annotations
from contextlib import contextmanager
import os
import numpy as np
from PySide6.QtGui import QOpenGLContext
from . import _gl as g

# Additional storage only (Qt's framebuffer is not included). Refuse rather
# than silently lowering MSAA or allocating unbounded DPR-dependent targets.
MAX_ATTACHMENT_BYTES = 256 * 1024 * 1024

_VERTEX = """#version 330 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
_COMPOSITE = """#version 330 core
uniform SAMPLER u_scene;
uniform SAMPLER u_accum;
uniform SAMPLER u_reveal;
uniform int u_samples;
out vec4 frag_colour;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 result = vec4(0.0);
    for (int s = 0; s < u_samples; ++s) {
        vec4 base = FETCH(u_scene, p, s);
        vec4 accum = FETCH(u_accum, p, s);
        float reveal = clamp(FETCH(u_reveal, p, s).r, 0.0, 1.0);
        vec3 colour = accum.rgb / max(accum.a, 1e-8);
        result += vec4(colour * (1.0 - reveal) + base.rgb * reveal,
                       (1.0 - reveal) + base.a * reveal);
    }
    frag_colour = result / float(u_samples);
}
"""

def weighted_fragment(source):
    """Reuse the exact surface/coverage shader, then change its output only."""
    source = source.replace('out vec4 frag_colour;', 'layout(location=0) out vec4 frag_colour;\nlayout(location=1) out vec4 reveal_colour;', 1)
    source = source.replace('void main()', 'void surface_colour()', 1)
    return source + """
uniform int u_oit_pass;
uniform float u_weight_scale;
void main() {
    surface_colour();
    float a = clamp(frag_colour.a, 0.0, 1.0);
    if (a <= 0.0) discard;
    if (u_oit_pass == 1) {
        frag_colour = vec4(a);
    } else {
        // Our orthographic projection uses reversed depth: near == 1.
        // Bounded weight, no perspective near/far or 1/z formula.
        float z = clamp(gl_FragCoord.z, 0.0, 1.0);
        float w = (0.1 + 0.9 * z * z * z) * u_weight_scale;
        frag_colour = vec4(frag_colour.rgb * a, a) * w;
        if (u_oit_pass == 2) reveal_colour = vec4(a);
    }
}
"""


@contextmanager
def preserved_state(native=None):
    """State touched by this component, including separate read/draw targets."""
    if native is not None:
        saved = native.save_state()
        try:yield
        finally:native.restore_state(saved)
        return
    ints = {name: int(g.glGetIntegerv(getattr(g, name))) for name in (
        'GL_DRAW_FRAMEBUFFER_BINDING', 'GL_READ_FRAMEBUFFER_BINDING',
        'GL_CURRENT_PROGRAM', 'GL_VERTEX_ARRAY_BINDING', 'GL_ACTIVE_TEXTURE',
        'GL_RENDERBUFFER_BINDING', 'GL_DEPTH_FUNC',
        'GL_BLEND_SRC_RGB', 'GL_BLEND_DST_RGB', 'GL_BLEND_SRC_ALPHA', 'GL_BLEND_DST_ALPHA',
        'GL_BLEND_EQUATION_RGB', 'GL_BLEND_EQUATION_ALPHA')}
    viewport = g.glGetIntegerv(g.GL_VIEWPORT).copy()
    depth_mask = bool(g.glGetBooleanv(g.GL_DEPTH_WRITEMASK))
    colour_mask = g.glGetBooleanv(g.GL_COLOR_WRITEMASK).copy()
    enables = {cap: bool(g.glIsEnabled(cap)) for cap in (
        g.GL_BLEND, g.GL_DEPTH_TEST, g.GL_MULTISAMPLE, g.GL_SCISSOR_TEST,
        g.GL_FRAMEBUFFER_SRGB, g.GL_CULL_FACE, g.GL_SAMPLE_ALPHA_TO_COVERAGE)}
    bindings = []
    for unit in range(4):
        g.glActiveTexture(g.GL_TEXTURE0 + unit)
        bindings.append(tuple(int(g.glGetIntegerv(target)) for target in (
            g.GL_TEXTURE_BINDING_2D, g.GL_TEXTURE_BINDING_2D_MULTISAMPLE, g.GL_TEXTURE_BINDING_BUFFER)))
    g.glActiveTexture(ints['GL_ACTIVE_TEXTURE'])
    try:
        yield
    finally:
        g.glBindFramebuffer(g.GL_DRAW_FRAMEBUFFER, ints['GL_DRAW_FRAMEBUFFER_BINDING'])
        g.glBindFramebuffer(g.GL_READ_FRAMEBUFFER, ints['GL_READ_FRAMEBUFFER_BINDING'])
        g.glViewport(*viewport)
        g.glUseProgram(ints['GL_CURRENT_PROGRAM'])
        g.glBindVertexArray(ints['GL_VERTEX_ARRAY_BINDING'])
        g.glBindRenderbuffer(g.GL_RENDERBUFFER, ints['GL_RENDERBUFFER_BINDING'])
        g.glDepthFunc(ints['GL_DEPTH_FUNC']); g.glDepthMask(depth_mask)
        g.glColorMask(*colour_mask)
        g.glBlendFuncSeparate(*(ints[name] for name in ('GL_BLEND_SRC_RGB','GL_BLEND_DST_RGB',
                                                      'GL_BLEND_SRC_ALPHA','GL_BLEND_DST_ALPHA')))
        g.glBlendEquationSeparate(ints['GL_BLEND_EQUATION_RGB'], ints['GL_BLEND_EQUATION_ALPHA'])
        for cap, enabled in enables.items():
            (g.glEnable if enabled else g.glDisable)(cap)
        for unit, values in enumerate(bindings):
            g.glActiveTexture(g.GL_TEXTURE0 + unit)
            for target, value in zip((g.GL_TEXTURE_2D, g.GL_TEXTURE_2D_MULTISAMPLE, g.GL_TEXTURE_BUFFER), values):
                g.glBindTexture(target, value)
        g.glActiveTexture(ints['GL_ACTIVE_TEXTURE'])


class WeightedOIT:
    def __init__(self,resources=None):
        self.resources=resources or g
        self.triangle_program = self.segment_program = self.vao = 0
        self.composite_programs = {}
        self.composite_uniforms = {}
        self.uniforms = {}
        self.textures = []; self.framebuffers = []; self.depth = 0
        self.size = None
        self.attachment_bytes = 0
        self.indexed = None
        self.single_pass = False
        self.single_pass_error = None
        self._indexed_checked = False
        self._prefer_single = os.environ.get('UNIT_CELL_GUI_WBOIT_BACKEND','auto').lower() != 'dual'

    def prepare_indexed(self):
        if self._indexed_checked:
            return
        self._indexed_checked = True
        context = QOpenGLContext.currentContext()
        if context is None:
            return
        fmt = context.format()
        core = (fmt.majorVersion(),fmt.minorVersion()) >= (4,0)
        if not core and not context.hasExtension(b'GL_ARB_draw_buffers_blend'):
            return
        suffix = '' if core else 'ARB'
        names = ('glGetIntegerv','glGetIntegeri_v','glGetBooleani_v','glIsEnabledi',
                 'glEnablei','glDisablei','glBlendFuncSeparatei'+suffix,
                 'glBlendEquationSeparatei'+suffix,'glColorMaski','glGetError')
        try:
            from ._ucg_native import GlIndexedBlend
            self.indexed = GlIndexedBlend([int(context.getProcAddress(n.encode('ascii')) or 0) for n in names])
        except Exception as exc:
            self.single_pass_error = str(exc) or type(exc).__name__

    @contextmanager
    def preserve_indexed(self):
        self.prepare_indexed()
        if self.indexed is None:
            yield
            return
        state = self.indexed.save()
        try:
            yield
        finally:
            self.indexed.restore(state)

    def initialize(self, create_program=None):
        if self.triangle_program:
            return
        # Import here to keep the surface shaders in their existing module.
        from .opengl_viewer import (_create_program, _PRIMITIVE_VERTEX_SHADER,
                                     _FRAGMENT_SHADER, _EDGE_FRAGMENT_SHADER, _EDGE_GEOMETRY_SHADER)
        if create_program is None:create_program = _create_program
        try:
            self.triangle_program = create_program(_PRIMITIVE_VERTEX_SHADER, weighted_fragment(_FRAGMENT_SHADER))
            self.segment_program = create_program(_PRIMITIVE_VERTEX_SHADER, weighted_fragment(_EDGE_FRAGMENT_SHADER),
                                                    _EDGE_GEOMETRY_SHADER)
            for multisample in (False, True):
                source = _COMPOSITE.replace('SAMPLER', 'sampler2DMS' if multisample else 'sampler2D')
                source = source.replace('FETCH', 'texelFetch')  # s == 0 for single-sample textures
                self.composite_programs[multisample] = create_program(_VERTEX, source)
                program = self.composite_programs[multisample]
                self.composite_uniforms[program] = {
                    name: g.glGetUniformLocation(program, name)
                    for name in ('u_scene', 'u_accum', 'u_reveal', 'u_samples')}
            self.vao = int(self.resources.glGenVertexArrays(1))
            names = ('u_projection','u_orientation','u_records','u_first','u_record_texels','u_use_order',
                     'u_viewport','u_edge_width','u_depth_bias','u_poly_opacities',
                     'u_poly_opacities_enabled','u_oit_pass','u_weight_scale')
            self.uniforms = {p: {name: g.glGetUniformLocation(p, name) for name in names}
                             for p in (self.triangle_program, self.segment_program)}
        except BaseException:
            self.cleanup()
            raise

    def release_attachments(self):
        if self.framebuffers:
            self.resources.glDeleteFramebuffers(len(self.framebuffers), self.framebuffers)
        if self.textures:
            self.resources.glDeleteTextures(len(self.textures), self.textures)
        if self.depth:
            self.resources.glDeleteRenderbuffers(1, [self.depth])
        self.textures = []; self.framebuffers = []; self.depth = 0
        self.size = None; self.attachment_bytes = 0; self.single_pass = False

    def ensure_size(self, width, height, samples):
        samples = max(1, int(samples))
        size = (int(width), int(height), samples)
        if size == self.size:
            return
        self.release_attachments()
        limit = min(int(g.glGetIntegerv(g.GL_MAX_TEXTURE_SIZE)),
                    int(g.glGetIntegerv(g.GL_MAX_RENDERBUFFER_SIZE)))
        if min(width,height) <= 0 or max(width,height) > limit:
            raise RuntimeError(f'WBOIT size {width} x {height} exceeds texture limit {limit}')
        required = width * height * samples * 20  # RGBA8 + depth24 (4) + RGBA16F + R32F
        if required > MAX_ATTACHMENT_BYTES:
            raise RuntimeError(f'WBOIT attachments require {required / 2**20:.1f} MiB; limit is {MAX_ATTACHMENT_BYTES / 2**20:.0f} MiB')
        if samples > 1 and samples > min(int(g.glGetIntegerv(g.GL_MAX_COLOR_TEXTURE_SAMPLES)),
                                         int(g.glGetIntegerv(g.GL_MAX_SAMPLES))):
            raise RuntimeError(f'WBOIT float textures do not support {samples} MSAA samples')
        target = g.GL_TEXTURE_2D_MULTISAMPLE if samples > 1 else g.GL_TEXTURE_2D
        try:
            # The state guard covers texture units 0..3. Never allocate on
            # an arbitrary host-selected texture unit outside that range.
            g.glActiveTexture(g.GL_TEXTURE0)
            self.depth = int(self.resources.glGenRenderbuffers(1))
            native=getattr(self.resources,'native',None)
            if native is not None:
                self.resources._check()
                actual=native.configure_depth(self.depth,width,height,samples)
                if actual!=samples:
                    raise RuntimeError(f'WBOIT depth has {actual} samples instead of {samples}')
            else:
                g.glBindRenderbuffer(g.GL_RENDERBUFFER, self.depth)
                if samples > 1:
                    g.glRenderbufferStorageMultisample(g.GL_RENDERBUFFER, samples, g.GL_DEPTH_COMPONENT24, width, height)
                    actual = int(g.glGetRenderbufferParameteriv(g.GL_RENDERBUFFER, g.GL_RENDERBUFFER_SAMPLES))
                    if actual != samples:
                        raise RuntimeError(f'WBOIT depth has {actual} samples instead of {samples}')
                else:
                    g.glRenderbufferStorage(g.GL_RENDERBUFFER, g.GL_DEPTH_COMPONENT24, width, height)
            for index, fmt in enumerate((g.GL_RGBA8, g.GL_RGBA16F, g.GL_R32F)):
                texture = int(self.resources.glGenTextures(1)); self.textures.append(texture)
                external = g.GL_RED if index == 2 else g.GL_RGBA
                if native is not None:
                    native.configure_color(texture,width,height,samples,int(fmt),int(external))
                else:
                    g.glBindTexture(target, texture)
                    if samples > 1:
                        g.glTexImage2DMultisample(target, samples, fmt, width, height, g.GL_TRUE)
                    else:
                        g.glTexImage2D(target, 0, fmt, width, height, 0, external, g.GL_FLOAT, None)
                        g.glTexParameteri(target, g.GL_TEXTURE_MIN_FILTER, g.GL_NEAREST)
                        g.glTexParameteri(target, g.GL_TEXTURE_MAG_FILTER, g.GL_NEAREST)
                        g.glTexParameteri(target, g.GL_TEXTURE_MAX_LEVEL, 0)
                fbo = int(self.resources.glGenFramebuffers(1)); self.framebuffers.append(fbo)
                if native is not None:
                    status=native.configure_framebuffer(fbo,texture,self.depth,samples)
                else:
                    g.glBindFramebuffer(g.GL_FRAMEBUFFER, fbo)
                    g.glFramebufferTexture2D(g.GL_FRAMEBUFFER, g.GL_COLOR_ATTACHMENT0, target, texture, 0)
                    g.glFramebufferRenderbuffer(g.GL_FRAMEBUFFER, g.GL_DEPTH_ATTACHMENT, g.GL_RENDERBUFFER, self.depth)
                    g.glDrawBuffer(g.GL_COLOR_ATTACHMENT0); g.glReadBuffer(g.GL_COLOR_ATTACHMENT0)
                    status = g.glCheckFramebufferStatus(g.GL_FRAMEBUFFER)
                if status != g.GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(f'WBOIT framebuffer {index} is incomplete: 0x{status:x}')
            self.size = size; self.attachment_bytes = required
            self.prepare_indexed()
            if self.indexed is not None and self._prefer_single:
                # Reuse the accumulation FBO; the revealage FBO remains available
                # for the two-pass fallback, with no additional attachment storage.
                g.glBindFramebuffer(g.GL_FRAMEBUFFER,self.framebuffers[1])
                try:
                    self._attach_revealage(self.textures[2])
                    g.glDrawBuffers([g.GL_COLOR_ATTACHMENT0,g.GL_COLOR_ATTACHMENT1])
                    status = g.glCheckFramebufferStatus(g.GL_FRAMEBUFFER)
                    if status != g.GL_FRAMEBUFFER_COMPLETE:
                        raise RuntimeError(f'combined WBOIT framebuffer incomplete: 0x{status:x}')
                    self.single_pass = True
                except Exception as exc:
                    self.single_pass_error = str(exc) or type(exc).__name__
                    self._disable_combined()
        except BaseException:
            self.release_attachments()
            raise

    def begin_scene(self, background):
        g.glBindFramebuffer(g.GL_FRAMEBUFFER, self.framebuffers[0])
        g.glViewport(0, 0, *self.size[:2])
        g.glDisable(g.GL_SCISSOR_TEST); g.glDisable(g.GL_FRAMEBUFFER_SRGB)
        g.glDisable(g.GL_CULL_FACE); g.glDisable(g.GL_SAMPLE_ALPHA_TO_COVERAGE)
        g.glColorMask(True,True,True,True)
        g.glEnable(g.GL_MULTISAMPLE); g.glEnable(g.GL_DEPTH_TEST)
        g.glDepthFunc(g.GL_GREATER); g.glDepthMask(True); g.glDisable(g.GL_BLEND)
        g.glClearBufferfv(g.GL_COLOR, 0, np.asarray(background, np.float32))
        g.glClearBufferfv(g.GL_DEPTH, 0, np.zeros(1, np.float32))

    def _begin(self, index, clear, source, destination):
        g.glBindFramebuffer(g.GL_FRAMEBUFFER, self.framebuffers[index])
        g.glViewport(0, 0, *self.size[:2])
        g.glClearBufferfv(g.GL_COLOR, 0, np.full(4, clear, np.float32))
        g.glEnable(g.GL_DEPTH_TEST); g.glDepthFunc(g.GL_GREATER); g.glDepthMask(False)
        g.glEnable(g.GL_BLEND); g.glBlendEquation(g.GL_FUNC_ADD); g.glBlendFunc(source,destination)

    def _attach_revealage(self,texture):
        native = getattr(self.resources,'native',None)
        if native is not None:
            self.resources._check()
            native.attach_revealage(self.framebuffers[1],texture,self.size[2])
        else:
            target = g.GL_TEXTURE_2D_MULTISAMPLE if self.size[2] > 1 else g.GL_TEXTURE_2D
            g.glBindFramebuffer(g.GL_FRAMEBUFFER,self.framebuffers[1])
            g.glFramebufferTexture2D(g.GL_FRAMEBUFFER,g.GL_COLOR_ATTACHMENT1,target,texture,0)

    def _disable_combined(self):
        self.single_pass = False
        self._prefer_single = False
        g.glBindFramebuffer(g.GL_FRAMEBUFFER,self.framebuffers[1])
        g.glDrawBuffer(g.GL_COLOR_ATTACHMENT0)
        self._attach_revealage(0)

    def begin_combined(self):
        if not self.single_pass:
            return False
        try:
            g.glBindFramebuffer(g.GL_FRAMEBUFFER,self.framebuffers[1])
            g.glViewport(0,0,*self.size[:2])
            g.glClearBufferfv(g.GL_COLOR,0,np.zeros(4,np.float32))
            g.glClearBufferfv(g.GL_COLOR,1,np.ones(4,np.float32))
            g.glEnable(g.GL_DEPTH_TEST); g.glDepthFunc(g.GL_GREATER); g.glDepthMask(False)
            self.indexed.configure()
            return True
        except Exception as exc:
            self.single_pass_error = str(exc) or type(exc).__name__
            self._disable_combined()
            return False

    def begin_accumulation(self):
        self._begin(1, 0., g.GL_ONE, g.GL_ONE)

    def begin_revealage(self):
        self._begin(2, 1., g.GL_ZERO, g.GL_ONE_MINUS_SRC_COLOR)

    def composite(self, qt_framebuffer, submission=None):
        g.glBindFramebuffer(g.GL_FRAMEBUFFER, qt_framebuffer)
        g.glViewport(0, 0, *self.size[:2])
        g.glDisable(g.GL_DEPTH_TEST); g.glDepthMask(False); g.glDisable(g.GL_BLEND)
        program = self.composite_programs[self.size[2] > 1]
        uniforms = self.composite_uniforms[program]
        g.glUseProgram(program); g.glBindVertexArray(self.vao)
        target = g.GL_TEXTURE_2D_MULTISAMPLE if self.size[2] > 1 else g.GL_TEXTURE_2D
        for unit, (name, texture) in enumerate(zip(('u_scene','u_accum','u_reveal'), self.textures)):
            g.glActiveTexture(g.GL_TEXTURE0 + unit); g.glBindTexture(target,texture)
            g.glUniform1i(uniforms[name],unit)
        g.glUniform1i(uniforms['u_samples'], self.size[2])
        if submission is not None:
            return submission.draw_arrays_batch([(self.vao,0,int(g.GL_TRIANGLES),0,3,-1)])
        g.glDrawArrays(g.GL_TRIANGLES, 0, 3)

    def cleanup(self):
        self.release_attachments()
        for program in (self.triangle_program, self.segment_program, *self.composite_programs.values()):
            if program:
                g.glDeleteProgram(program)
        if self.vao:
            self.resources.glDeleteVertexArrays(1, [self.vao])
        self.triangle_program = self.segment_program = self.vao = 0
        self.composite_programs.clear(); self.composite_uniforms.clear(); self.uniforms.clear()
