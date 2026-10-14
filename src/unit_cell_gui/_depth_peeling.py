"""OpenGL 3.3 front-to-back peeling of resident face/edge instances.

All fragments use the renderer's reversed depth convention (near = larger).
There is no fixed layer cutoff: an occlusion query ends an exhausted stack.
Two-times supersampling avoids dependence on optional sample-shading features.
"""
from OpenGL import GL
import numpy as np

_SCREEN_VERTEX = '''#version 330 core
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
'''
_ACCUMULATE_FRAGMENT = '''#version 330 core
uniform sampler2D u_layer;
out vec4 frag_colour;
void main() {
    frag_colour = texelFetch(u_layer, ivec2(gl_FragCoord.xy), 0);
}
'''
_RESOLVE_FRAGMENT = '''#version 330 core
uniform sampler2D u_base;
uniform sampler2D u_accum;
out vec4 frag_colour;
vec3 combined(ivec2 p) {
    vec4 accumulated = texelFetch(u_accum, p, 0);
    return accumulated.rgb + accumulated.a * texelFetch(u_base, p, 0).rgb;
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) * 2;
    vec3 colour = combined(p) + combined(p + ivec2(1, 0))
        + combined(p + ivec2(0, 1)) + combined(p + ivec2(1, 1));
    frag_colour = vec4(colour * 0.25, 1.0);
}
'''
_PEEL_DECLARATIONS = '''
uniform sampler2D u_opaque_depth;
uniform sampler2D u_previous_depth;
uniform int u_first_layer;
void reject_previous_layers() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float z = gl_FragCoord.z;
    if (z <= texelFetch(u_opaque_depth, pixel, 0).r) discard;
    if (u_first_layer == 0 && z >= texelFetch(u_previous_depth, pixel, 0).r) discard;
}
'''


def peeling_fragment(original):
    """Retain material/line coverage, then emit premultiplied layer colour."""
    result = original.replace('void main() {', _PEEL_DECLARATIONS + '\nvoid main() {\n    reject_previous_layers();', 1)
    first, last = result.rsplit('\n}', 1)
    return first + '\n    frag_colour.rgb *= frag_colour.a;\n}' + last


class DepthPeeler:
    scale = 2

    def __init__(self, owner, compile_program, primitive_vertex, face_fragment, edge_fragment, edge_geometry):
        self.owner = owner
        self.programs = []
        self.targets = []
        self.size = None
        self.query = 0
        self.vao = 0
        self.layers = 0
        try:
            self.face_program = compile_program(primitive_vertex, peeling_fragment(face_fragment))
            self.programs.append(self.face_program)
            self.edge_program = compile_program(primitive_vertex, peeling_fragment(edge_fragment), edge_geometry)
            self.programs.append(self.edge_program)
            self.accumulate_program = compile_program(_SCREEN_VERTEX, _ACCUMULATE_FRAGMENT)
            self.programs.append(self.accumulate_program)
            self.resolve_program = compile_program(_SCREEN_VERTEX, _RESOLVE_FRAGMENT)
            self.programs.append(self.resolve_program)
            self.vao = int(GL.glGenVertexArrays(1))
            self.query = int(GL.glGenQueries(1)[0])
            names = ('u_projection', 'u_orientation', 'u_light_dir', 'u_records', 'u_first',
                     'u_record_texels', 'u_use_order', 'u_viewport', 'u_edge_width', 'u_depth_bias',
                     'u_opaque_depth', 'u_previous_depth', 'u_first_layer')
            names += ('u_poly_opacities', 'u_poly_opacities_enabled')
            for program in (self.face_program, self.edge_program):
                owner._instance_uniforms[program] = {name: GL.glGetUniformLocation(program, name) for name in names}
            self.screen_uniforms = {program: {name: GL.glGetUniformLocation(program, name) for name in
                ('u_layer', 'u_base', 'u_accum')} for program in (self.accumulate_program, self.resolve_program)}
        except Exception:
            self.close()
            raise

    def _texture(self, width, height, internal, format):
        texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, format, GL.GL_FLOAT, None)
        for name in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, name, GL.GL_NEAREST)
        for name in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, name, GL.GL_CLAMP_TO_EDGE)
        return texture

    def _target(self, width, height, internal, depth=False):
        target = dict(fbo=int(GL.glGenFramebuffers(1)), colour=0, depth=0)
        self.targets.append(target)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target['fbo'])
        target['colour'] = self._texture(width, height, internal, GL.GL_RGBA)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                 GL.GL_TEXTURE_2D, target['colour'], 0)
        if depth:
            target['depth'] = self._texture(width, height, GL.GL_DEPTH_COMPONENT32F, GL.GL_DEPTH_COMPONENT)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                     GL.GL_TEXTURE_2D, target['depth'], 0)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Depth-peeling framebuffer is incomplete')
        return target

    def release_targets(self):
        for target in self.targets:
            textures = [value for key, value in target.items() if key != 'fbo' and value]
            if textures:
                GL.glDeleteTextures(len(textures), textures)
            GL.glDeleteFramebuffers(1, [target['fbo']])
        self.targets = []
        self.size = None
        self.layers = 0

    def begin(self, width, height, background):
        self.qt_fbo = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        self.qt_viewport = tuple(int(value) for value in GL.glGetIntegerv(GL.GL_VIEWPORT))
        size = (width * self.scale, height * self.scale)
        try:
            if self.size != size:
                self.release_targets()
                GL.glActiveTexture(GL.GL_TEXTURE0)
                self.base = self._target(*size, GL.GL_RGBA8, depth=True)
                self.peels = [self._target(*size, GL.GL_RGBA16F, depth=True) for _ in range(2)]
                self.accum = self._target(*size, GL.GL_RGBA16F)
                self.size = size
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.base['fbo'])
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glViewport(0, 0, *size)
            GL.glClearColor(background.redF(), background.greenF(), background.blueF(), 1)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self.owner._render_scale = self.scale
        except Exception:
            self.restore()
            self.release_targets()
            raise

    @staticmethod
    def _bind_texture(unit, texture):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)

    def _draw_ranges(self, program, texels, ranges, default):
        owner = self.owner
        owner._set_instance_uniforms(program, texels, width_px=1.575, depth_bias=1e-5)
        uniforms = owner._instance_uniforms[program]
        GL.glUniform1i(uniforms['u_opaque_depth'], 1)
        GL.glUniform1i(uniforms['u_previous_depth'], 2)
        GL.glUniform1i(uniforms['u_first_layer'], int(self.layers == 0))
        # Adjacent records can be drawn together: peeling needs no CPU sorting.
        runs = []
        for item in ranges:
            prefix = item.buffer_key or default
            if runs and runs[-1][0] == prefix and runs[-1][1] + runs[-1][2] == item.first:
                runs[-1][2] += item.count
            else:
                runs.append([prefix, item.first, item.count])
        for prefix, first, count in runs:
            owner._bind_primitive(getattr(owner, prefix + '_vao'), getattr(owner, prefix + '_texture'))
            owner._draw_primitive_instances(GL.GL_TRIANGLES if texels == 4 else GL.GL_LINES, first, count, program=program)
        GL.glBindVertexArray(0)

    def finish(self, faces, edges):
        self.layers = 0
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.accum['fbo'])
        GL.glClearColor(0, 0, 0, 1)  # RGB sum, remaining transmittance
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        # Fullscreen accumulation is restricted to the conservative projected
        # polyhedron bounds; atoms/base colour still retain the full viewport.
        indices = {item.poly_index for item in faces + edges if item.poly_index is not None}
        vertices = [self.owner._display_scene.polyhedra[i].vertices for i in indices]
        if vertices:
            camera = np.concatenate(vertices) @ np.asarray(self.owner.orientation).T
            ratio = self.owner.devicePixelRatioF() * self.scale
            xy = (camera[:,:2] * self.owner._screen_scale() +
                  [self.owner.width()/2 + self.owner.pan_x, self.owner.height()/2 - self.owner.pan_y]) * ratio
            low = np.clip(np.floor(xy.min(axis=0)-4*ratio),0,self.size).astype(int)
            high = np.clip(np.ceil(xy.max(axis=0)+4*ratio),0,self.size).astype(int)
            extent = np.maximum(0,high-low)
            GL.glScissor(int(low[0]),int(low[1]),int(extent[0]),int(extent[1]))
            GL.glEnable(GL.GL_SCISSOR_TEST)
        while True:
            current = self.peels[self.layers % 2]
            previous = self.peels[1 - self.layers % 2]
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, current['fbo'])
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
            GL.glClearColor(0, 0, 0, 0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._bind_texture(1, self.base['depth'])
            self._bind_texture(2, previous['depth'])
            GL.glBeginQuery(GL.GL_ANY_SAMPLES_PASSED, self.query)
            try:
                self._draw_ranges(self.face_program, 4, faces, '_face')
                self._draw_ranges(self.edge_program, 3, edges, '_face_edge')
            finally:
                GL.glEndQuery(GL.GL_ANY_SAMPLES_PASSED)
            if not int(GL.glGetQueryObjectuiv(self.query, GL.GL_QUERY_RESULT)):
                break
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.accum['fbo'])
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFuncSeparate(GL.GL_DST_ALPHA, GL.GL_ONE, GL.GL_ZERO, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glUseProgram(self.accumulate_program)
            self._bind_texture(1, current['colour'])
            GL.glUniform1i(self.screen_uniforms[self.accumulate_program]['u_layer'], 1)
            GL.glBindVertexArray(self.vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            self.layers += 1
        self.restore()
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glUseProgram(self.resolve_program)
        self._bind_texture(1, self.base['colour'])
        self._bind_texture(2, self.accum['colour'])
        GL.glUniform1i(self.screen_uniforms[self.resolve_program]['u_base'], 1)
        GL.glUniform1i(self.screen_uniforms[self.resolve_program]['u_accum'], 2)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        self._unbind_textures()
        self.owner._restore_render_state()

    def restore(self):
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.qt_fbo)
        GL.glViewport(*self.qt_viewport)
        self.owner._render_scale = 1

    def _unbind_textures(self):
        for unit in (0, 1, 2):
            self._bind_texture(unit, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def close(self):
        self.release_targets()
        if self.query:
            GL.glDeleteQueries(1, [self.query])
            self.query = 0
        if self.vao:
            GL.glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        for program in self.programs:
            self.owner._instance_uniforms.pop(program, None)
            GL.glDeleteProgram(program)
        self.programs = []
