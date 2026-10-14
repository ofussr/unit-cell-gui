"""Cached OpenGL 3.3 unit-cell renderer.

Static meshes stay in GPU buffers when visibility or orientation changes.
Atom silhouettes use a shared instanced ring, oriented in the vertex shader.
View-dependent hatching is cached per polyhedron and density.
"""

from __future__ import annotations

import ctypes
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_TEXTURE_BUFFER,
    GL_MAX_TEXTURE_BUFFER_SIZE,
    GL_TEXTURE0,
    GL_RGBA32F,
    GL_R32F,
    GL_INT,
    GL_DYNAMIC_DRAW,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_GEOMETRY_SHADER,
    GL_GREATER,
    GL_LINES,
    GL_LINK_STATUS,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_INT,
    GL_VERTEX_SHADER,
    glAttachShader,
    glActiveTexture,
    glBindTexture,
    glTexBuffer,
    glGenTextures,
    glDeleteTextures,
    glUniform1i,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glBufferSubData,
    glClear,
    glClearColor,
    glClearDepth,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteShader,
    glDeleteVertexArrays,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glDrawArrays,
    glDrawArraysInstanced,
    glDrawElementsInstanced,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetIntegerv,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniform2f,
    glUniform3f,
    glUniformMatrix3fv,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glVertexAttribIPointer,
    glVertexAttribDivisor,
    glViewport,
)
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPainter, QPen, QPolygonF
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy

from .gl_geometry import indexed_cylinder_mesh, indexed_sphere_mesh, line_vertices, polygon_triangles
from .hatching import hatch_division_count
from .models import DisplayOptions, Scene, StyleOverrides
from ._scene_updates import SceneView, prepare, same_polyhedron
from ._depth_peeling import DepthPeeler
from .qt_config import configure_qt_opengl, preferred_surface_format
from .view_helpers import _polyhedron_hatches, screen_drag_orientation


# ``viewer.py`` imports this module while normal hosts are still constructing
# their application. Configure Qt here so legacy ``CrystalCanvas`` imports get
# the safe multi-widget setup without requiring a new bootstrap call.
configure_qt_opengl()


# One static vertex format is shared by every buffer:
# position.xyz, normal.xyz, colour.rgba, lit
_VERTEX_FLOATS = 11
_VERTEX_STRIDE = _VERTEX_FLOATS * 4

_ATOM_MAX_COMPONENTS = 8
# The base sphere uses position.xyz and normal.xyz once per mesh vertex.
_ATOM_MESH_STRIDE = 6 * 4
# Each atom stores center.xyz, radius, eight RGB colours and eight thresholds.
_ATOM_INSTANCE_FLOATS = 4 + 3 * _ATOM_MAX_COMPONENTS + _ATOM_MAX_COMPONENTS
_ATOM_INSTANCE_STRIDE = _ATOM_INSTANCE_FLOATS * 4

# Shared ring vertices store solid.xy and dashed.xy. Instances store
# center.xyz, radius, colour.rgba, partial-occupancy flag.
# Half-bond frame: start.xyz, radial_u.xyz, radial_v.xyz, finish.xyz, RGBA.
_BOND_INSTANCE_FLOATS = 16
# Texture-buffer records use whole RGBA texels; positions have unused W.
_TRIANGLE_INSTANCE_FLOATS = 16
_SEGMENT_INSTANCE_FLOATS = 12
_OUTLINE_SEGMENTS = 64
_OUTLINE_INSTANCE_FLOATS = 9
_OUTLINE_INSTANCE_STRIDE = _OUTLINE_INSTANCE_FLOATS * 4

_CELL_DASH_PATTERN = (8.0, 4.0)


@dataclass(frozen=True)
class _PolyhedronRange:
    site_key: str
    first: int
    count: int
    center: np.ndarray | None = None
    poly_index: int | None = None
    buffer_key: str | None = None


@dataclass
class _HatchBuffer:
    vao: int
    vbo: int
    count: int = 0
    geometry_key: tuple | None = None


# Keep a few depth-density variants for each polyhedron at the current view.
# Hiding a depth-extreme object and showing it again can reuse both variants,
# without retaining an unbounded history of rotations or slider positions.
_HATCH_VARIANTS_PER_POLYHEDRON = 4


_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_colour;
layout(location = 3) in float a_lit;

uniform mat4 u_projection;
uniform mat3 u_orientation;

out vec3 v_normal;
out vec4 v_colour;
out float v_lit;

void main() {
    vec3 camera_position = u_orientation * a_position;
    v_normal = u_orientation * a_normal;
    v_colour = a_colour;
    v_lit = a_lit;
    gl_Position = u_projection * vec4(camera_position, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec4 v_colour;
in float v_lit;

uniform vec3 u_light_dir;
out vec4 frag_colour;

void main() {
    float normal_length = length(v_normal);
    float diffuse = normal_length > 1e-7
        ? max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0)
        : 1.0;
    float lighting = mix(1.0, 0.34 + 0.66 * diffuse, v_lit);
    frag_colour = vec4(v_colour.rgb * lighting, v_colour.a);
}
"""


_ATOM_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_colour0;
layout(location = 3) in vec3 a_colour1;
layout(location = 4) in vec3 a_colour2;
layout(location = 5) in vec3 a_colour3;
layout(location = 6) in vec3 a_colour4;
layout(location = 7) in vec3 a_colour5;
layout(location = 8) in vec3 a_colour6;
layout(location = 9) in vec3 a_colour7;
layout(location = 10) in vec4 a_cumulative0;
layout(location = 11) in vec4 a_cumulative1;

layout(location = 12) in vec4 a_center_radius;

uniform mat4 u_projection;
uniform mat3 u_orientation;

out vec3 v_normal;
flat out vec3 v_colour0;
flat out vec3 v_colour1;
flat out vec3 v_colour2;
flat out vec3 v_colour3;
flat out vec3 v_colour4;
flat out vec3 v_colour5;
flat out vec3 v_colour6;
flat out vec3 v_colour7;
flat out vec4 v_cumulative0;
flat out vec4 v_cumulative1;

void main() {
    vec3 camera_position = u_orientation * (a_center_radius.xyz + a_position * a_center_radius.w);
    v_normal = u_orientation * a_normal;
    v_colour0 = a_colour0;
    v_colour1 = a_colour1;
    v_colour2 = a_colour2;
    v_colour3 = a_colour3;
    v_colour4 = a_colour4;
    v_colour5 = a_colour5;
    v_colour6 = a_colour6;
    v_colour7 = a_colour7;
    v_cumulative0 = a_cumulative0;
    v_cumulative1 = a_cumulative1;
    gl_Position = a_center_radius.w < 0.0 ? vec4(2.0, 2.0, 2.0, 1.0)
        : u_projection * vec4(camera_position, 1.0);
}
"""

_ATOM_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
flat in vec3 v_colour0;
flat in vec3 v_colour1;
flat in vec3 v_colour2;
flat in vec3 v_colour3;
flat in vec3 v_colour4;
flat in vec3 v_colour5;
flat in vec3 v_colour6;
flat in vec3 v_colour7;
flat in vec4 v_cumulative0;
flat in vec4 v_cumulative1;

uniform vec3 u_light_dir;
uniform float u_engraving;
out vec4 frag_colour;

void main() {
    vec3 normal = normalize(v_normal);

    // Camera-facing occupancy pie.  Because v_normal is already in camera
    // space, the sector layout stays fixed on screen while the crystal turns.
    float tau = 6.283185307179586;
    float pie_position = fract(atan(normal.x, normal.y) / tau + 1.0);
    vec3 base = vec3(1.0);  // unoccupied / hidden occupancy is white
    if      (pie_position < v_cumulative0.x) base = v_colour0;
    else if (pie_position < v_cumulative0.y) base = v_colour1;
    else if (pie_position < v_cumulative0.z) base = v_colour2;
    else if (pie_position < v_cumulative0.w) base = v_colour3;
    else if (pie_position < v_cumulative1.x) base = v_colour4;
    else if (pie_position < v_cumulative1.y) base = v_colour5;
    else if (pie_position < v_cumulative1.z) base = v_colour6;
    else if (pie_position < v_cumulative1.w) base = v_colour7;

    if (u_engraving > 0.5) {
        // Match the old engraving renderer: flat monochrome atoms, no glossy
        // highlight, no directional shading.  The outline is a separate
        // depth-tested OpenGL pass.
        frag_colour = vec4(base, 1.0);
        return;
    }

    vec3 light_dir = normalize(u_light_dir);
    float diffuse = max(dot(normal, light_dir), 0.0);
    vec3 view_dir = vec3(0.0, 0.0, 1.0);
    vec3 half_dir = normalize(light_dir + view_dir);
    float specular = pow(max(dot(normal, half_dir), 0.0), 34.0);

    // Coloured mode keeps the glossy 3-D styling.
    float frontness = max(normal.z, 0.0);
    float rim = mix(0.58, 1.0, smoothstep(0.0, 0.22, frontness));
    float lighting = (0.46 + 0.54 * diffuse) * rim;
    vec3 shaded = base * lighting + vec3(0.24 * specular);
    frag_colour = vec4(clamp(shaded, 0.0, 1.0), 1.0);
}
"""

_BOND_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_start;
layout(location = 2) in vec3 a_u_radius;
layout(location = 3) in vec3 a_v_radius;
layout(location = 4) in vec3 a_finish;
layout(location = 5) in vec4 a_colour;
uniform mat4 u_projection;
uniform mat3 u_orientation;
out vec3 v_normal;
out vec4 v_colour;
out float v_lit;
void main() {
    vec3 radial = a_position.x * a_u_radius + a_position.y * a_v_radius;
    // Explicit endpoint coordinates keep both coloured halves coincident.
    vec3 world_position = mix(a_start, a_finish, a_position.z) + radial;
    v_normal = u_orientation * (radial / max(length(a_u_radius), 1e-12));
    v_colour = a_colour;
    v_lit = 1.0;
    gl_Position = length(a_u_radius) <= 1e-12 ? vec4(2.0, 2.0, 2.0, 1.0)
        : u_projection * vec4(u_orientation * world_position, 1.0);
}
"""

# One shared selector mesh supplies corners 0/1/2 for triangles or 0/1
# for lines. A uniform selects a resident instance range on OpenGL 3.3,
# preserving face/edge transparency order without uploading sorted geometry.
_PRIMITIVE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in float a_corner;
layout(location = 1) in int a_record;
uniform int u_use_order;
uniform samplerBuffer u_records;
uniform samplerBuffer u_poly_opacities;
uniform int u_poly_opacities_enabled;
uniform int u_first;
uniform int u_record_texels;
uniform mat4 u_projection;
uniform mat3 u_orientation;
out vec3 v_normal;
out vec4 v_colour;
out float v_lit;
void main() {
    int record = u_use_order != 0 ? a_record - u_first : u_first + gl_InstanceID;
    int base = record * u_record_texels;
    vec3 position = texelFetch(u_records, base + int(a_corner)).xyz;
    v_colour = texelFetch(u_records, base + u_record_texels - 1);
    if (u_poly_opacities_enabled != 0) {
        int material = int(texelFetch(u_records, base).w);
        if (material > 0) v_colour.a = texelFetch(u_poly_opacities, material - 1).r;
    }
    v_normal = vec3(0.0);
    v_lit = 0.0;
    gl_Position = u_projection * vec4(u_orientation * position, 1.0);
}
"""


_EDGE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 2) in vec4 a_colour;

uniform mat4 u_projection;
uniform mat3 u_orientation;

out vec4 v_colour;
uniform float u_opacity;

void main() {
    vec3 camera_position = u_orientation * a_position;
    v_colour = a_colour;
    v_colour.a *= u_opacity;
    gl_Position = u_projection * vec4(camera_position, 1.0);
}
"""


_ATOM_OUTLINE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec4 a_ring;
layout(location = 1) in vec4 a_center_radius;
layout(location = 2) in vec4 a_colour;
layout(location = 3) in float a_partial;

uniform mat4 u_projection;
uniform mat3 u_orientation;
uniform float u_outline_padding;

out vec4 v_colour;

void main() {
    vec3 camera_position = u_orientation * a_center_radius.xyz;
    vec2 ring = mix(a_ring.xy, a_ring.zw, a_partial);
    camera_position.xy += ring * (a_center_radius.w + u_outline_padding);
    v_colour = a_colour;
    gl_Position = a_center_radius.w < 0.0 ? vec4(2.0, 2.0, 2.0, 1.0)
        : u_projection * vec4(camera_position, 1.0);
}
"""


_EDGE_GEOMETRY_SHADER = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 v_colour[];
out vec4 g_colour;
noperspective out vec2 g_line_position;
flat out float g_line_length;
flat out float g_half_width;

uniform vec2 u_viewport;
uniform float u_edge_width;
uniform float u_depth_bias;

void emit_edge_vertex(
    vec4 clip,
    vec2 ndc_offset,
    vec4 colour,
    vec2 line_position,
    float line_length,
    float half_width
) {
    vec4 shifted = clip;
    shifted.xy += ndc_offset * clip.w;
    shifted.z += u_depth_bias * clip.w;
    gl_Position = shifted;
    g_colour = colour;
    g_line_position = line_position;
    g_line_length = line_length;
    g_half_width = half_width;
    EmitVertex();
}

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 n0 = p0.xy / p0.w;
    vec2 n1 = p1.xy / p1.w;
    vec2 delta_pixels = vec2(
        (n1.x - n0.x) * 0.5 * u_viewport.x,
        (n1.y - n0.y) * 0.5 * u_viewport.y
    );
    float length_pixels = length(delta_pixels);
    if (length_pixels < 1e-5) {
        return;
    }
    vec2 direction_pixels = delta_pixels / length_pixels;
    vec2 perpendicular_pixels = vec2(-direction_pixels.y, direction_pixels.x);
    float half_width = 0.5 * u_edge_width;
    float feather = 1.25;
    float outer_half_width = half_width + feather;
    float outer_end_extension = half_width + feather;

    // Expand by half the requested width at both ends so neighbouring edge
    // quads overlap, then add a one-pixel feather for shader antialiasing.
    vec2 end_extend_ndc = vec2(
        direction_pixels.x * outer_end_extension * 2.0 / u_viewport.x,
        direction_pixels.y * outer_end_extension * 2.0 / u_viewport.y
    );
    vec2 side_ndc = vec2(
        perpendicular_pixels.x * outer_half_width * 2.0 / u_viewport.x,
        perpendicular_pixels.y * outer_half_width * 2.0 / u_viewport.y
    );

    vec2 start = -end_extend_ndc;
    vec2 finish = end_extend_ndc;
    emit_edge_vertex(
        p0, start + side_ndc, v_colour[0],
        vec2(-outer_end_extension, outer_half_width), length_pixels, half_width
    );
    emit_edge_vertex(
        p0, start - side_ndc, v_colour[0],
        vec2(-outer_end_extension, -outer_half_width), length_pixels, half_width
    );
    emit_edge_vertex(
        p1, finish + side_ndc, v_colour[1],
        vec2(length_pixels + outer_end_extension, outer_half_width),
        length_pixels, half_width
    );
    emit_edge_vertex(
        p1, finish - side_ndc, v_colour[1],
        vec2(length_pixels + outer_end_extension, -outer_half_width),
        length_pixels, half_width
    );
    EndPrimitive();
}
"""


_EDGE_FRAGMENT_SHADER = """
#version 330 core
in vec4 g_colour;
noperspective in vec2 g_line_position;
flat in float g_line_length;
flat in float g_half_width;
out vec4 frag_colour;

void main() {
    float outside_x = max(
        -g_half_width - g_line_position.x,
        g_line_position.x - (g_line_length + g_half_width)
    );
    float outside_y = abs(g_line_position.y) - g_half_width;
    vec2 outside = max(vec2(outside_x, outside_y), vec2(0.0));
    float coverage = 1.0 - smoothstep(0.0, 1.25, length(outside));
    if (coverage <= 0.0) {
        discard;
    }
    frag_colour = vec4(g_colour.rgb, g_colour.a * coverage);
}
"""



def _compile_shader(kind: int, source: str) -> int:
    shader = glCreateShader(kind)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        glDeleteShader(shader)
        raise RuntimeError(f"OpenGL shader compilation failed:\n{log}")
    return shader


def _create_program(
    vertex_source: str = _VERTEX_SHADER,
    fragment_source: str = _FRAGMENT_SHADER,
    geometry_source: str | None = None,
) -> int:
    shaders = [
        _compile_shader(GL_VERTEX_SHADER, vertex_source),
        _compile_shader(GL_FRAGMENT_SHADER, fragment_source),
    ]
    if geometry_source is not None:
        shaders.append(_compile_shader(GL_GEOMETRY_SHADER, geometry_source))
    program = glCreateProgram()
    for shader in shaders:
        glAttachShader(program, shader)
    glLinkProgram(program)
    for shader in shaders:
        glDeleteShader(shader)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        glDeleteProgram(program)
        raise RuntimeError(f"OpenGL shader linking failed:\n{log}")
    return program


def _colour4(colour: QColor, alpha: float = 1.0) -> np.ndarray:
    return np.asarray(
        [colour.redF(), colour.greenF(), colour.blueF(), float(alpha)],
        dtype=np.float32,
    )


def _engraving_colour(element: str, colour_value: str) -> QColor:
    if element == "O":
        return QColor(248, 248, 248)
    colour = QColor(colour_value)
    grey = int(round(0.299 * colour.red() + 0.587 * colour.green() + 0.114 * colour.blue()))
    grey = max(55, min(220, grey))
    return QColor(grey, grey, grey)


def _blend_components(components) -> QColor:
    weighted = np.zeros(3, dtype=float)
    total = 0.0
    for component in components:
        colour = QColor(component.colour)
        weight = max(0.0, float(component.occupancy))
        weighted += weight * np.asarray((colour.red(), colour.green(), colour.blue()))
        total += weight
    if total <= 1e-12:
        return QColor("#b0b0b0")
    red, green, blue = np.clip(np.rint(weighted / total), 0, 255).astype(int)
    return QColor(int(red), int(green), int(blue))


def _polyhedron_edge_colour(fill: QColor) -> QColor:
    return QColor(fill).lighter(145)


def _vertex_block(
    positions: np.ndarray,
    normals: np.ndarray,
    colour: QColor,
    *,
    alpha: float = 1.0,
    lit: bool = True,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    if positions.shape != normals.shape:
        raise ValueError("positions and normals must have matching shapes")
    if positions.shape[0] == 0:
        return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
    colour_values = np.broadcast_to(_colour4(colour, alpha), (positions.shape[0], 4))
    lit_values = np.full((positions.shape[0], 1), 1.0 if lit else 0.0, dtype=np.float32)
    return np.ascontiguousarray(
        np.concatenate((positions, normals, colour_values, lit_values), axis=1),
        dtype=np.float32,
    )


def _combine_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    nonempty = [block for block in blocks if block.size]
    if not nonempty:
        return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(nonempty, axis=0), dtype=np.float32)


class OpenGLUnitCellViewer(QOpenGLWidget):
    """Depth-buffered renderer implementing the public viewer API."""

    orientation_changed = Signal(object)
    interaction_finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFormat(preferred_surface_format())

        self.scene: Scene | None = None
        self._scene_updates: SceneView | None = None
        self._pending_atoms: set[int] = set()
        self._pending_bonds: set[int] = set()
        self._pending_polyhedra: set[int] = set()
        self._pending_opacities: set[int] = set()
        self._poly_opacity_data = np.empty((0,1), np.float32)
        self._classified_poly_ranges = None
        self._poly_opacity_vbo = 0
        self._poly_opacity_texture = 0
        self._pending_sections: set[int] = set()
        self._base_ranges_by_poly: dict[str, dict] = {}
        self._partial_poly_gpu: dict[int, dict] = {}
        self._partial_section_gpu: dict[int, _HatchBuffer] = {}
        self._poly_revisions: dict[int, int] = {}
        self._atom_gpu_rows: dict[int, int] = {}
        self._outline_gpu_rows: dict[int, int] = {}
        self._bond_gpu_rows: dict[int, int] = {}
        self._atom_instance_data = np.empty((0, _ATOM_INSTANCE_FLOATS), np.float32)
        self._outline_instance_data = np.empty((0, _OUTLINE_INSTANCE_FLOATS), np.float32)
        self._bond_instance_data = np.empty((0, _BOND_INSTANCE_FLOATS), np.float32)
        self._engraving_instance_data = np.empty((0, _SEGMENT_INSTANCE_FLOATS), np.float32)
        self.orientation = np.eye(3)
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.show_atoms = True
        self.show_external_atoms = True
        self.show_bonds = True
        self.show_cell = True
        self.show_basis = True
        self.show_polyhedra = False
        self.engraving = False
        self.hatching = True
        self.hatch_density = 22
        self.hatch_grip = 50
        self.atom_scale = 1.0
        self.rotation_enabled = True
        self.empty_text = "No structure is loaded"
        self.title_text = ""
        self.atom_component_visibility: dict[str, bool] = {}
        self.atom_component_colours: dict[str, str] = {}
        self.polyhedron_site_visibility: dict[str, bool] = {}
        self.polyhedron_site_colours: dict[str, str] = {}
        self.polyhedron_opaque_sites: set[str] = set()
        self._last_mouse: QPointF | None = None
        self._drag_button = None

        self._program = 0
        self._uniform_projection = -1
        self._uniform_orientation = -1
        self._uniform_light = -1
        self._atom_program = 0
        self._atom_uniform_projection = -1
        self._atom_uniform_orientation = -1
        self._atom_uniform_light = -1
        self._atom_uniform_engraving = -1
        self._edge_program = 0
        self._edge_uniform_projection = -1
        self._edge_uniform_orientation = -1
        self._edge_uniform_viewport = -1
        self._edge_uniform_width = -1
        self._edge_uniform_depth_bias = -1
        self._edge_uniform_opacity = -1
        self.opengl_error: str | None = None
        self._cleanup_context = None
        self._cleaning_up = False
        self._depth_peeler = None
        self._render_scale = 1

        self._atom_vao = 0
        self._atom_vbo = 0
        self._atom_mesh_vbo = 0
        self._atom_ebo = 0
        self._atom_index_count = 0
        self._atom_count = 0
        self._opaque_vao = 0
        self._opaque_vbo = 0
        self._opaque_count = 0
        self._opaque_ranges: list[_PolyhedronRange] = []
        self._bond_vao = 0
        self._bond_vbo = 0
        self._bond_mesh_vbo = 0
        self._bond_ebo = 0
        self._bond_index_count = 0
        self._bond_program = 0
        self._triangle_program = 0
        self._segment_program = 0
        self._primitive_mesh_vbo = 0
        self._instance_uniforms: dict[int, dict[str, int]] = {}
        self._primitive_chunks: dict[str, list[tuple[int, int, int]]] = {}
        self._primitive_texel_capacity = 65536
        self._bound_primitive_chunks: list[tuple[int, int, int]] = []
        self._bound_instance_texture = 0
        self._opaque_texture = 0
        self._face_texture = 0
        self._face_edge_texture = 0
        self._engraving_bond_texture = 0
        self._bond_count = 0
        self._bond_dirty = True
        self._cell_vao = 0
        self._cell_vbo = 0
        self._cell_count = 0
        self._face_vao = 0
        self._face_vbo = 0
        self._face_count = 0
        self._face_ranges: list[_PolyhedronRange] = []
        self._face_edge_vao = 0
        self._face_edge_vbo = 0
        self._face_edge_order_vbo = 0
        self._face_edge_order_capacity = 0
        self._face_edge_order_records: np.ndarray | None = None
        self._face_edge_count = 0
        self._face_edge_ranges: list[_PolyhedronRange] = []
        self._hatch_buffers: dict[int, _HatchBuffer] = {}
        self._hatch_draw_indices: tuple[int, ...] = ()
        self._hatch_buffers_need_clear = False
        self._hatch_count = 0
        self._hatch_dirty = True
        self._atom_outline_vao = 0
        self._atom_outline_vbo = 0
        self._atom_outline_mesh_vbo = 0
        self._atom_outline_program = 0
        self._atom_outline_uniforms: dict[str, int] = {}
        self._atom_outline_count = 0
        self._atom_outline_dirty = True
        self._intersection_vao = 0
        self._intersection_vbo = 0
        self._intersection_count = 0
        self._intersection_ranges: list[_PolyhedronRange] = []
        self._engraving_bond_vao = 0
        self._engraving_bond_vbo = 0
        self._engraving_bond_count = 0
        self._gpu_dirty = True
        self._atom_dirty = True
        self._bond_dirty = True
        self._intersection_dirty = True

        # Face triangulation depends on the Scene; reusable cylinders and
        # triangles stay resident on the GPU.
        self._poly_face_mesh_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._hatch_mesh_cache: dict[int, OrderedDict[int, np.ndarray]] = {}
        self._hatch_view_key: tuple | None = None

        self._show_fps = False
        self._frame_times: deque[float] = deque(maxlen=90)

        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    @property
    def _display_scene(self):
        if self._scene_updates is not None and self._scene_updates.base is self.scene:
            return self._scene_updates
        return self.scene

    def _global_atom_visible(self, index: int) -> bool:
        atom = self._display_scene.atoms[index]
        return self._atom_visible(atom) and (self.show_external_atoms or not atom.external)

    def _locally_visible(self, index: int) -> bool:
        return self._scene_updates is None or self._scene_updates.visibility_overrides.get(index, True)

    def get_atom_positions(self) -> np.ndarray:
        """Return an independent copy of effective Cartesian coordinates."""
        return (self._display_scene.atom_centers.copy() if self.scene is not None
                else np.empty((0, 3), dtype=float))

    def get_atom_visibility(self) -> np.ndarray:
        """Return the effective mask including host display/component switches."""
        return np.asarray([self.show_atoms and self._global_atom_visible(i) and self._locally_visible(i)
                           for i in range(len(self.scene.atoms))], dtype=bool) if self.scene is not None else np.zeros(0, bool)

    def get_polyhedra(self) -> tuple:
        """Return independently owned copies of effective polyhedra."""
        return tuple(replace(poly) for poly in self._display_scene.polyhedra) if self.scene is not None else ()

    def get_polyhedron_opacities(self) -> np.ndarray:
        """Independent effective material opacities, separate from visibility."""
        return np.asarray([self._polyhedron_opacity(i) for i in range(len(self.scene.polyhedra))],
                          dtype=float) if self.scene is not None else np.empty(0, float)

    def set_polyhedron_opacities(self, opacities) -> None:
        """Set indexed opacities in [0,1]; None restores a per-object default."""
        self.update_scene(polyhedron_opacities=opacities)

    def set_atom_positions(self, positions) -> None:
        self.update_scene(atom_positions=positions)

    def set_atom_visibility(self, visibility) -> None:
        self.update_scene(atom_visibility=visibility)

    def update_polyhedra(self, polyhedra) -> None:
        self.update_scene(polyhedra=polyhedra)

    def update_scene(self, *, atom_positions=None, atom_visibility=None, polyhedra=None, polyhedron_opacities=None) -> None:
        """Atomically merge indexed overrides, without replacing Scene or camera.

        Only supplied indices change. Positions are absolute Cartesian XYZ.
        Apply on the GUI thread, as with other viewer methods. GPU uploads are
        deferred to the next paint with the viewer's context current.
        """
        if self.scene is None:
            if atom_positions is None and atom_visibility is None and polyhedra is None and polyhedron_opacities is None:
                return
            raise RuntimeError('set_scene() must precede partial updates')
        positions, visibility, polys, opacities = prepare(self.scene, atom_positions, atom_visibility, polyhedra, polyhedron_opacities)
        effective = self._display_scene
        positions = {i: value for i, value in positions.items() if not np.array_equal(value, effective.atom_centers[i])}
        visibility = {i: value for i, value in visibility.items() if value != self._locally_visible(i)}
        polys = {i: value for i, value in polys.items() if not same_polyhedron(value, effective.polyhedra[i])}
        current_opacities = self._scene_updates.opacity_overrides if self._scene_updates else {}
        opacities = {i: value for i, value in opacities.items() if current_opacities.get(i) != value}
        if not positions and not visibility and not polys and not opacities:
            return
        if self._scene_updates is None:
            self._scene_updates = SceneView(self.scene)
        view = self._scene_updates
        for index, value in opacities.items():
            if value is None:
                view.opacity_overrides.pop(index, None)
            else:
                view.opacity_overrides[index] = value
        if opacities:
            self._classified_poly_ranges = None
            self._hatch_dirty = True
        self._pending_opacities.update(set(opacities) | set(polys))
        atoms = set(positions) | set(visibility)
        points = []
        for index in atoms:
            radius = self._atom_radius(self.scene.atoms[index])
            if self._global_atom_visible(index) and (self._locally_visible(index) or visibility.get(index, self._locally_visible(index))):
                points.extend(((view.atom_centers[index].copy(), radius),
                               (positions.get(index, view.atom_centers[index]), radius)))
            self._pending_bonds.update(view.bonds_by_atom[index])
        self._pending_sections.update(view.sections_near(points))
        for index, value in positions.items():
            view.atom_centers[index] = value
            if np.array_equal(value, self.scene.atom_centers[index]):
                view.position_overrides.pop(index, None)
            else:
                view.position_overrides[index] = value
        for index, value in visibility.items():
            if value:
                view.visibility_overrides.pop(index, None)
            else:
                view.visibility_overrides[index] = False
        for index, value in polys.items():
            if same_polyhedron(value, self.scene.polyhedra[index]):
                view.polyhedron_overrides.pop(index, None)
                view.polyhedra[index] = self.scene.polyhedra[index]
            else:
                view.polyhedron_overrides[index] = value
                view.polyhedra[index] = value
            for key in tuple(self._poly_face_mesh_cache):
                if key[0] == index:
                    del self._poly_face_mesh_cache[key]
            self._hatch_mesh_cache.pop(index, None)
            self._poly_revisions[index] = self._poly_revisions.get(index, 0) + 1
        if polys:
            view.update_bounds(polys)
            self._hatch_dirty = True
        self._pending_atoms.update(atoms)
        self._pending_polyhedra.update(polys)
        self._pending_sections.update(polys)
        self.update()

    def reset_scene_updates(self) -> None:
        """Restore original positions/polyhedra and remove local hiding."""
        view = self._scene_updates
        if view is None:
            return
        self.update_scene(atom_positions={i: self.scene.atom_centers[i] for i in view.position_overrides},
                          atom_visibility={i: True for i in view.visibility_overrides},
                          polyhedra={i: self.scene.polyhedra[i] for i in view.polyhedron_overrides},
                          polyhedron_opacities={i: None for i in view.opacity_overrides})
        self._scene_updates = None

    def sizeHint(self) -> QSize:
        return QSize(760, 620)

    def set_debug_fps(self, enabled: bool = True) -> None:
        self._show_fps = bool(enabled)
        self._frame_times.clear()
        self.update()

    def _mark_gpu_dirty(self) -> None:
        """Compatibility helper: mark every GPU representation dirty."""
        self._gpu_dirty = True
        self._atom_dirty = True
        self._bond_dirty = True
        self._intersection_dirty = True
        self._hatch_dirty = True
        self._atom_outline_dirty = True
        self.update()

    def _clear_mesh_caches(self) -> None:
        self._poly_face_mesh_cache.clear()
        self._hatch_mesh_cache.clear()
        self._hatch_view_key = None

    def set_scene(self, scene: Scene | None, *, reset_camera: bool = True) -> None:
        if scene is not None and not isinstance(scene, Scene):
            raise TypeError("scene must be a unit_cell_gui.Scene or None")
        self.scene = scene
        self._scene_updates = None
        self._pending_atoms.clear(); self._pending_bonds.clear()
        self._pending_polyhedra.clear(); self._pending_sections.clear()
        self._pending_opacities.clear()
        self._classified_poly_ranges = None
        self._poly_revisions.clear()
        self._atom_gpu_rows.clear(); self._outline_gpu_rows.clear(); self._bond_gpu_rows.clear()
        self._clear_mesh_caches()
        self._hatch_buffers_need_clear = True
        if reset_camera:
            self.zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
        self._mark_gpu_dirty()

    def set_orientation(self, orientation: np.ndarray, *, emit: bool = False) -> None:
        matrix = np.asarray(orientation, dtype=float).reshape(3, 3).copy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError("orientation must contain finite values")
        self.orientation = matrix
        self._hatch_dirty = True
        self.update()
        if emit:
            self.orientation_changed.emit(self.orientation.copy())

    def set_display_options(self, options: DisplayOptions | None = None, **changes) -> None:
        values = {
            name: getattr(options, name) if options is not None else getattr(self, name)
            for name in DisplayOptions.__dataclass_fields__
        }
        unknown = set(changes).difference(values)
        if unknown:
            raise TypeError(f"Unknown display options: {', '.join(sorted(unknown))}")
        values.update(changes)
        normalized = DisplayOptions(**values)
        old = {name: getattr(self, name) for name in DisplayOptions.__dataclass_fields__}
        for name in DisplayOptions.__dataclass_fields__:
            setattr(self, name, getattr(normalized, name))
        changed = {name for name in old if old[name] != getattr(self, name)}
        if not changed:
            self.update()
            return

        # Overlay / interaction-only switches must never rebuild millions of
        # sphere/cylinder vertices.
        if changed & {"hatching", "hatch_density", "hatch_grip", "engraving"}:
            self._hatch_dirty = True
        if "show_atoms" in changed:
            self._atom_outline_dirty = True
        if "engraving" in changed:
            self._gpu_dirty = True
            self._intersection_dirty = True
        if changed & {"engraving", "show_external_atoms"}:
            self._atom_dirty = True
            self._atom_outline_dirty = True
        if "show_external_atoms" in changed:
            self._bond_dirty = True
        if "atom_scale" in changed:
            self._atom_dirty = True
            self._atom_outline_dirty = True
            self._intersection_dirty = True
        # Intersection colours depend on colour/BW mode. Build the layer
        # lazily only when both atoms and polyhedra are actually visible.
        if "show_external_atoms" in changed:
            self._intersection_dirty = True
        self.update()

    def set_labels(self, *, empty: str | None = None, title: str | None = None) -> None:
        if empty is not None:
            self.empty_text = str(empty)
        if title is not None:
            self.title_text = str(title)
        self.update()

    def set_style_overrides(self, overrides: StyleOverrides | None = None) -> None:
        normalized = overrides or StyleOverrides()
        if not isinstance(normalized, StyleOverrides):
            raise TypeError("overrides must be unit_cell_gui.StyleOverrides or None")

        new_atom_visibility = dict(normalized.atom_component_visibility)
        new_atom_colours = dict(normalized.atom_component_colours)
        new_poly_visibility = dict(normalized.polyhedron_site_visibility)
        new_poly_colours = dict(normalized.polyhedron_site_colours)
        new_poly_opaque = set(normalized.polyhedron_opaque_sites)

        atom_visibility_changed = new_atom_visibility != self.atom_component_visibility
        atom_colours_changed = new_atom_colours != self.atom_component_colours
        poly_visibility_changed = new_poly_visibility != self.polyhedron_site_visibility
        poly_colours_changed = new_poly_colours != self.polyhedron_site_colours
        poly_style_changed = (
            poly_colours_changed
            or new_poly_opaque != self.polyhedron_opaque_sites
        )

        self.atom_component_visibility = new_atom_visibility
        self.atom_component_colours = new_atom_colours
        self.polyhedron_site_visibility = new_poly_visibility
        self.polyhedron_site_colours = new_poly_colours
        self.polyhedron_opaque_sites = new_poly_opaque

        if atom_visibility_changed or atom_colours_changed:
            self._atom_dirty = True
            self._bond_dirty = True
            self._atom_outline_dirty = True
        if atom_visibility_changed:
            self._intersection_dirty = True
        if poly_colours_changed:
            self._intersection_dirty = True
        if poly_style_changed:
            self._gpu_dirty = True
        if poly_visibility_changed:
            # Face/edge/intersection buffers contain all polyhedra. Visibility
            # is applied at draw time, without changing their vertex data.
            # Hatching still needs a cheap density check: the visible depth
            # extrema may have changed, but unchanged meshes stay cached.
            self._hatch_dirty = True
        self.update()

    def zoom_by(self, steps: float) -> None:
        self.zoom = float(np.clip(self.zoom * (1.12 ** float(steps)), 0.25, 4.5))
        self._hatch_dirty = True
        self.update()

    def _component_visible(self, component) -> bool:
        return self.atom_component_visibility.get(component.key, True)

    def _component_colour(self, component) -> QColor:
        base = self.atom_component_colours.get(component.key, component.colour)
        if self.engraving:
            return _engraving_colour(component.element, base)
        return QColor(base)

    def _atom_visible(self, atom) -> bool:
        return any(self._component_visible(component) for component in atom.components)

    def _atom_radius(self, atom) -> float:
        weighted = 0.0
        total = 0.0
        for component in atom.components:
            occupancy = max(0.0, float(component.occupancy))
            weighted += occupancy * float(component.radius)
            total += occupancy
        radius = weighted / total if total > 1e-12 else 0.22
        return radius * self.atom_scale

    def _mixed_atom_colour(self, atom) -> QColor:
        components = [
            component
            for component in atom.components
            if self._component_visible(component) and component.occupancy > 0.0
        ]
        total = sum(float(component.occupancy) for component in components)
        if total <= 1e-12:
            return QColor("#8b8b8b")
        channels = np.zeros(3, dtype=float)
        for component in components:
            colour = self._component_colour(component)
            channels += float(component.occupancy) * np.asarray(
                (colour.red(), colour.green(), colour.blue()), dtype=float
            )
        red, green, blue = np.clip(np.rint(channels / total), 0, 255).astype(int)
        return QColor(int(red), int(green), int(blue))

    def _dominant_atom_colour(self, atom) -> QColor:
        visible = [
            component for component in atom.components
            if self._component_visible(component) and component.occupancy > 0.0
        ]
        if not visible:
            return QColor("#8b8b8b")
        maximum = max(float(component.occupancy) for component in visible)
        dominant = [
            component for component in visible
            if abs(float(component.occupancy) - maximum) < 1e-12
        ]
        if len(dominant) == 1:
            return self._component_colour(dominant[0])
        colours = [self._component_colour(component) for component in dominant]
        return QColor(
            int(round(sum(colour.red() for colour in colours) / len(colours))),
            int(round(sum(colour.green() for colour in colours) / len(colours))),
            int(round(sum(colour.blue() for colour in colours) / len(colours))),
        )

    def _dominant_atom_colour_coloured(self, atom) -> QColor:
        """Dominant atom colour independent of the current engraving switch."""
        visible = [
            component for component in atom.components
            if self._component_visible(component) and component.occupancy > 0.0
        ]
        if not visible:
            return QColor("#8b8b8b")
        maximum = max(float(component.occupancy) for component in visible)
        dominant = [
            component for component in visible
            if abs(float(component.occupancy) - maximum) < 1e-12
        ]
        colours = [
            QColor(self.atom_component_colours.get(component.key, component.colour))
            for component in dominant
        ]
        if len(colours) == 1:
            return colours[0]
        return QColor(
            int(round(sum(colour.red() for colour in colours) / len(colours))),
            int(round(sum(colour.green() for colour in colours) / len(colours))),
            int(round(sum(colour.blue() for colour in colours) / len(colours))),
        )

    def _atom_pie_material(self, atom) -> tuple[np.ndarray, np.ndarray]:
        colours = np.ones((_ATOM_MAX_COMPONENTS, 3), dtype=np.float32)
        cumulative = np.zeros((_ATOM_MAX_COMPONENTS,), dtype=np.float32)

        if self.engraving:
            colour = self._mixed_atom_colour(atom)
            rgb = np.asarray([colour.redF(), colour.greenF(), colour.blueF()], dtype=np.float32)
            colours[0] = rgb
            cumulative[:] = 1.0
            return colours, cumulative

        raw = []
        for component in atom.components:
            occupancy = max(0.0, float(component.occupancy))
            if occupancy <= 1e-12:
                continue
            colour = self._component_colour(component) if self._component_visible(component) else QColor(255, 255, 255)
            rgb = np.asarray([colour.redF(), colour.greenF(), colour.blueF()], dtype=np.float32)
            raw.append((occupancy, rgb))

        if not raw:
            cumulative[:] = 1.0
            return colours, cumulative

        total = sum(item[0] for item in raw)
        scale = 1.0 / total if total > 1.0 else 1.0
        slots: list[tuple[float, np.ndarray]] = []
        for occupancy, rgb in raw:
            amount = occupancy * scale
            if len(slots) < _ATOM_MAX_COMPONENTS:
                slots.append((amount, rgb))
            else:
                previous_amount, previous_rgb = slots[-1]
                combined = previous_amount + amount
                if combined > 1e-12:
                    previous_rgb = (previous_amount * previous_rgb + amount * rgb) / combined
                slots[-1] = (combined, previous_rgb)

        running = 0.0
        for index, (amount, rgb) in enumerate(slots):
            running = min(1.0, running + amount)
            colours[index] = rgb
            cumulative[index] = running
        if slots:
            cumulative[len(slots):] = running
        return colours, cumulative

    def _projection_matrix(self) -> np.ndarray:
        width, height = max(1, self.width()), max(1, self.height())
        radius = self._display_scene.base_radius if self._display_scene is not None else 1.0
        scale = self._screen_scale()
        depth_extent = max(4.0 * radius, 1.0)
        return np.asarray(
            [
                [2.0 * scale / width, 0.0, 0.0, 2.0 * self.pan_x / width],
                [0.0, 2.0 * scale / height, 0.0, -2.0 * self.pan_y / height],
                [0.0, 0.0, 1.0 / depth_extent, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def _screen_scale(self) -> float:
        width, height = max(1, self.width()), max(1, self.height())
        radius = self._display_scene.base_radius if self._display_scene is not None else 1.0
        return 0.88 * min(width, height) / (2.0 * radius) * self.zoom

    def _project_camera_point(self, point: np.ndarray) -> QPointF:
        width, height = max(1, self.width()), max(1, self.height())
        scale = self._screen_scale()
        camera = np.asarray(point, dtype=float).reshape(3)
        return QPointF(
            width * 0.5 + float(camera[0]) * scale + self.pan_x,
            height * 0.5 - float(camera[1]) * scale + self.pan_y,
        )

    def _project_world_point(self, point: np.ndarray) -> QPointF:
        return self._project_camera_point(np.asarray(self.orientation, dtype=float) @ np.asarray(point, dtype=float))

    def _unproject_to_face(
        self,
        point: QPointF,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
    ) -> np.ndarray | None:
        width, height = max(1, self.width()), max(1, self.height())
        radius = self._display_scene.base_radius if self._display_scene is not None else 1.0
        scale = 0.88 * min(width, height) / (2.0 * radius) * self.zoom
        if scale <= 1e-12:
            return None
        x = (float(point.x()) - width * 0.5 - self.pan_x) / scale
        y = (height * 0.5 + self.pan_y - float(point.y())) / scale
        normal = np.asarray(plane_normal, dtype=float).reshape(3)
        origin = np.asarray(plane_point, dtype=float).reshape(3)
        if abs(float(normal[2])) <= 1e-12:
            return None
        z = float(origin[2]) - (
            float(normal[0]) * (x - float(origin[0]))
            + float(normal[1]) * (y - float(origin[1]))
        ) / float(normal[2])
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def _create_buffer() -> tuple[int, int]:
        vao = int(glGenVertexArrays(1))
        vbo = int(glGenBuffers(1))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, _VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, _VERTEX_STRIDE, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, _VERTEX_STRIDE, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, _VERTEX_STRIDE, ctypes.c_void_p(10 * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao, vbo

    def _create_atom_buffer(self) -> None:
        # Assign handles immediately so partial initialization can be cleaned up.
        self._atom_vao = int(glGenVertexArrays(1))
        self._atom_mesh_vbo = int(glGenBuffers(1))
        self._atom_ebo = int(glGenBuffers(1))
        self._atom_vbo = int(glGenBuffers(1))
        glBindVertexArray(self._atom_vao)
        positions, normals, indices = indexed_sphere_mesh()
        mesh = np.ascontiguousarray(np.concatenate((positions, normals), axis=1))
        glBindBuffer(GL_ARRAY_BUFFER, self._atom_mesh_vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh, GL_STATIC_DRAW)
        for location, offset in ((0, 0), (1, 3)):
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(
                location, 3, GL_FLOAT, GL_FALSE, _ATOM_MESH_STRIDE, ctypes.c_void_p(offset * 4)
            )
            glVertexAttribDivisor(location, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._atom_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        self._atom_index_count = int(indices.size)

        glBindBuffer(GL_ARRAY_BUFFER, self._atom_vbo)
        attributes = [(12, 4, 0)]
        attributes += [(location, 3, 4 + 3 * (location - 2)) for location in range(2, 10)]
        attributes += [(10, 4, 28), (11, 4, 32)]
        for location, size, offset in attributes:
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(
                location, size, GL_FLOAT, GL_FALSE, _ATOM_INSTANCE_STRIDE, ctypes.c_void_p(offset * 4)
            )
            glVertexAttribDivisor(location, 1)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    @staticmethod
    def _upload_static(vao: int, vbo: int, vertices: np.ndarray) -> int:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return int(vertices.shape[0])

    def _create_atom_outline_buffer(self) -> None:
        self._atom_outline_vao = int(glGenVertexArrays(1))
        self._atom_outline_mesh_vbo = int(glGenBuffers(1))
        self._atom_outline_vbo = int(glGenBuffers(1))
        glBindVertexArray(self._atom_outline_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._atom_outline_mesh_vbo)
        mesh = self._atom_outline_mesh()
        glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glVertexAttribDivisor(0, 0)
        glBindBuffer(GL_ARRAY_BUFFER, self._atom_outline_vbo)
        for location, size, offset in ((1, 4, 0), (2, 4, 4), (3, 1, 8)):
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE,
                                  _OUTLINE_INSTANCE_STRIDE, ctypes.c_void_p(offset * 4))
            glVertexAttribDivisor(location, 1)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _create_bond_buffer(self) -> None:
        self._bond_vao = int(glGenVertexArrays(1))
        self._bond_mesh_vbo = int(glGenBuffers(1))
        self._bond_ebo = int(glGenBuffers(1))
        self._bond_vbo = int(glGenBuffers(1))
        vertices, indices = indexed_cylinder_mesh()
        self._bond_index_count = len(indices)
        glBindVertexArray(self._bond_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._bond_mesh_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glVertexAttribDivisor(0, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._bond_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._bond_vbo)
        for location, size, offset in ((1, 3, 0), (2, 3, 3), (3, 3, 6), (4, 3, 9), (5, 4, 12)):
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, 64, ctypes.c_void_p(offset * 4))
            glVertexAttribDivisor(location, 1)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _create_primitive_buffer(self, prefix: str) -> None:
        vao = int(glGenVertexArrays(1))
        setattr(self, prefix + '_vao', vao)
        vbo = int(glGenBuffers(1))
        setattr(self, prefix + '_vbo', vbo)
        texture = int(glGenTextures(1))
        setattr(self, prefix + '_texture', texture)
        self._primitive_chunks[prefix] = [(vbo, texture, 0)]
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._primitive_mesh_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
        glVertexAttribDivisor(0, 0)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, 16, None, GL_STATIC_DRAW)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, texture)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, vbo)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        if prefix.endswith('_face_edge'):
            setattr(self, prefix + '_order_vbo', int(glGenBuffers(1)))
            setattr(self, prefix + '_order_capacity', 0)
            setattr(self, prefix + '_order_records', None)
            glBindBuffer(GL_ARRAY_BUFFER, getattr(self, prefix + '_order_vbo'))
            glBufferData(GL_ARRAY_BUFFER, 4, None, GL_DYNAMIC_DRAW)
            glEnableVertexAttribArray(1)
            glVertexAttribIPointer(1, 1, GL_INT, 4, ctypes.c_void_p(0))
            glVertexAttribDivisor(1, 1)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def initializeGL(self) -> None:
        context = self.context()
        if context is not None and context is not self._cleanup_context:
            self._disconnect_context_cleanup()
            context.aboutToBeDestroyed.connect(self._context_about_to_be_destroyed)
            self._cleanup_context = context
        self._reset_gpu_objects()
        try:
            self._primitive_texel_capacity = int(glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE))
            self._program = _create_program()
            self._atom_program = _create_program(_ATOM_VERTEX_SHADER, _ATOM_FRAGMENT_SHADER)
            self._edge_program = _create_program(
                _EDGE_VERTEX_SHADER,
                _EDGE_FRAGMENT_SHADER,
                _EDGE_GEOMETRY_SHADER,
            )
            self._atom_outline_program = _create_program(
                _ATOM_OUTLINE_VERTEX_SHADER, _EDGE_FRAGMENT_SHADER, _EDGE_GEOMETRY_SHADER,
            )
            self._atom_outline_uniforms = {
                name: glGetUniformLocation(self._atom_outline_program, name)
                for name in ("u_projection", "u_orientation", "u_outline_padding",
                             "u_viewport", "u_edge_width", "u_depth_bias")
            }
            self._bond_program = _create_program(_BOND_VERTEX_SHADER, _FRAGMENT_SHADER)
            self._triangle_program = _create_program(_PRIMITIVE_VERTEX_SHADER, _FRAGMENT_SHADER)
            self._segment_program = _create_program(
                _PRIMITIVE_VERTEX_SHADER, _EDGE_FRAGMENT_SHADER, _EDGE_GEOMETRY_SHADER,
            )
            uniform_names = ('u_projection', 'u_orientation', 'u_light_dir', 'u_records',
                             'u_first', 'u_record_texels', 'u_use_order', 'u_viewport', 'u_edge_width', 'u_depth_bias', 'u_poly_opacities', 'u_poly_opacities_enabled')
            self._instance_uniforms = {
                program: {name: glGetUniformLocation(program, name) for name in uniform_names}
                for program in (self._bond_program, self._triangle_program, self._segment_program)
            }
            self._poly_opacity_vbo = int(glGenBuffers(1))
            self._poly_opacity_texture = int(glGenTextures(1))
            self._primitive_mesh_vbo = int(glGenBuffers(1))
            glBindBuffer(GL_ARRAY_BUFFER, self._primitive_mesh_vbo)
            corners = np.array([0., 1., 2.], dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, corners.nbytes, corners, GL_STATIC_DRAW)
            self._create_atom_buffer()
            self._create_primitive_buffer('_opaque')
            self._create_bond_buffer()
            self._cell_vao, self._cell_vbo = self._create_buffer()
            self._create_primitive_buffer('_face')
            self._create_primitive_buffer('_face_edge')
            self._create_atom_outline_buffer()
            self._intersection_vao, self._intersection_vbo = self._create_buffer()
            self._create_primitive_buffer('_engraving_bond')
            self._uniform_projection = glGetUniformLocation(self._program, "u_projection")
            self._uniform_orientation = glGetUniformLocation(self._program, "u_orientation")
            self._uniform_light = glGetUniformLocation(self._program, "u_light_dir")
            self._atom_uniform_projection = glGetUniformLocation(self._atom_program, "u_projection")
            self._atom_uniform_orientation = glGetUniformLocation(self._atom_program, "u_orientation")
            self._atom_uniform_light = glGetUniformLocation(self._atom_program, "u_light_dir")
            self._atom_uniform_engraving = glGetUniformLocation(self._atom_program, "u_engraving")
            self._edge_uniform_projection = glGetUniformLocation(self._edge_program, "u_projection")
            self._edge_uniform_orientation = glGetUniformLocation(self._edge_program, "u_orientation")
            self._edge_uniform_viewport = glGetUniformLocation(self._edge_program, "u_viewport")
            self._edge_uniform_width = glGetUniformLocation(self._edge_program, "u_edge_width")
            self._edge_uniform_depth_bias = glGetUniformLocation(self._edge_program, "u_depth_bias")
            self._edge_uniform_opacity = glGetUniformLocation(self._edge_program, "u_opacity")
            self._restore_render_state()
            self._gpu_dirty = True
            self._atom_dirty = True
            self._bond_dirty = True
            self._intersection_dirty = True
            self.opengl_error = None
        except Exception as exc:
            self.opengl_error = str(exc)

    def resizeGL(self, width: int, height: int) -> None:
        glViewport(0, 0, max(1, int(width)), max(1, int(height)))
        self._hatch_dirty = True

    def _restore_render_state(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glClearDepth(0.0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)

    def _cached_poly_face_mesh(self, poly_index: int, face_index: int, polyhedron, face):
        key = (int(poly_index), int(face_index))
        cached = self._poly_face_mesh_cache.get(key)
        if cached is None:
            world = np.asarray(polyhedron.vertices[list(face)], dtype=float)
            positions, normals = polygon_triangles(world, polyhedron.center)
            cached = (world, positions, normals)
            self._poly_face_mesh_cache[key] = cached
        return cached

    def _build_atom_instances(self) -> np.ndarray:
        """Prepare one compact record per visible atom, with no tessellation."""
        if self._display_scene is None:
            return np.empty((0, _ATOM_INSTANCE_FLOATS), dtype=np.float32)
        instances = np.empty((len(self._display_scene.atoms), _ATOM_INSTANCE_FLOATS), dtype=np.float32)
        count = 0
        for atom_index, (atom, center) in enumerate(zip(self._display_scene.atoms, self._display_scene.atom_centers)):
            if not self._atom_visible(atom):
                continue
            if atom.external and not self.show_external_atoms:
                continue
            colours, cumulative = self._atom_pie_material(atom)
            instances[count, :3] = center
            instances[count, 3] = self._atom_radius(atom) if self._locally_visible(atom_index) else -1.0
            instances[count, 4:28] = colours.ravel()
            instances[count, 28:] = cumulative
            count += 1
        return np.ascontiguousarray(instances[:count])

    def _selected_bond_ids(self, bond_indices=None) -> list[int]:
        if self._display_scene is None:
            return []
        indices = range(len(self._display_scene.bonds)) if bond_indices is None else bond_indices
        return [int(i) for i in indices if all(self._global_atom_visible(int(a)) for a in self._display_scene.bonds[int(i)])]

    def _build_colour_bond_instances(self, bond_indices=None) -> np.ndarray:
        ids = self._selected_bond_ids(bond_indices)
        if not ids:
            return np.empty((0, _BOND_INSTANCE_FLOATS), dtype=np.float32)
        bonds = self._display_scene.bonds[ids]
        a, b = self._display_scene.atom_centers[bonds[:, 0]], self._display_scene.atom_centers[bonds[:, 1]]
        midpoints = 0.5 * (a + b)
        starts = np.stack((a, midpoints), axis=1).reshape(-1, 3)
        finishes = np.stack((midpoints, b), axis=1).reshape(-1, 3)
        delta = finishes - starts
        lengths = np.linalg.norm(delta, axis=1)
        valid = lengths > 1e-12
        axes = np.zeros_like(delta)
        axes[valid] = delta[valid] / lengths[valid, None]
        references = np.zeros_like(axes); references[:, 2] = 1.0
        references[np.abs(axes[:, 2]) > 0.9] = (0., 1., 0.)
        u = np.cross(axes, references)
        u[valid] /= np.linalg.norm(u[valid], axis=1)[:, None]
        v = np.cross(axes, u)
        v[valid] /= np.linalg.norm(v[valid], axis=1)[:, None]
        local = np.repeat([all(self._locally_visible(int(i)) for i in pair) for pair in bonds], 2)
        u[~local] = 0; v[~local] = 0
        radius = max(0.018, 0.020 * float(self._display_scene.base_radius)) / 3.0
        records = np.empty((len(starts), _BOND_INSTANCE_FLOATS), dtype=np.float32)
        records[:, :3], records[:, 3:6], records[:, 6:9] = starts, radius * u, radius * v
        records[:, 9:12] = finishes
        colours = {int(i): _colour4(self._dominant_atom_colour_coloured(self._display_scene.atoms[int(i)]))
                   for i in np.unique(bonds)}
        records[:, 12:16] = [colours[int(i)] for i in bonds.ravel()]
        return records

    @staticmethod
    def _triangle_instances(positions: np.ndarray, colour: QColor, alpha: float) -> np.ndarray:
        triangles = np.asarray(positions, dtype=np.float32).reshape(-1, 3, 3)
        records = np.zeros((len(triangles), _TRIANGLE_INSTANCE_FLOATS), dtype=np.float32)
        for corner in range(3):
            records[:, corner * 4:corner * 4 + 3] = triangles[:, corner]
        records[:, 12:16] = _colour4(colour, alpha)
        return records

    @staticmethod
    def _segment_instance(first, second, colour: QColor, alpha: float = 1.0) -> np.ndarray:
        record = np.zeros((1, _SEGMENT_INSTANCE_FLOATS), dtype=np.float32)
        record[0, :3], record[0, 4:7] = first, second
        record[0, 8:12] = _colour4(colour, alpha)
        return record

    @staticmethod
    def _combine_instances(blocks: list[np.ndarray], width: int) -> np.ndarray:
        return np.concatenate(blocks, axis=0) if blocks else np.empty((0, width), dtype=np.float32)

    def _polyhedron_colour(self, polyhedron) -> QColor:
        if self.engraving:
            return QColor(245, 245, 245)
        return QColor(self.polyhedron_site_colours.get(
            polyhedron.site_key, _blend_components(polyhedron.components).name(),
        ))

    def _polyhedron_opacity(self, index, *, use_local=True) -> float:
        poly = self._display_scene.polyhedra[index]
        value = self._scene_updates.opacity_overrides.get(index) if self._scene_updates and use_local else None
        if self.engraving:
            return 0.0 if value == 0 else 1.0
        if value is not None:
            return value
        if poly.site_key in self.polyhedron_opaque_sites:
            return 1.0
        occupancy = min(1.0, sum(max(0.0, float(c.occupancy)) for c in poly.components))
        return 0.28 + 0.36 * occupancy

    def _build_static_geometry(self, *, poly_indices=None, include_cell=True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[_PolyhedronRange], np.ndarray, list[_PolyhedronRange], list[_PolyhedronRange]]:
        if self._display_scene is None:
            empty = np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
            atom_empty = np.empty((0, _ATOM_INSTANCE_FLOATS), dtype=np.float32)
            return (atom_empty, np.empty((0, _TRIANGLE_INSTANCE_FLOATS), np.float32), empty,
                    np.empty((0, _TRIANGLE_INSTANCE_FLOATS), np.float32), [],
                    np.empty((0, _SEGMENT_INSTANCE_FLOATS), np.float32), [], [])

        opaque_blocks: list[np.ndarray] = []
        opaque_ranges: list[_PolyhedronRange] = []
        cell_blocks: list[np.ndarray] = []
        face_blocks: list[np.ndarray] = []
        face_ranges: list[_PolyhedronRange] = []
        face_edge_blocks: list[np.ndarray] = []
        face_edge_ranges: list[_PolyhedronRange] = []

        if self.show_cell and include_cell:
            colour = QColor(90, 90, 90) if not self.engraving else QColor(60, 60, 60)
            for first, second in self._display_scene.cell_edges:
                positions, normals = line_vertices(
                    self._display_scene.cell_vertices[int(first)],
                    self._display_scene.cell_vertices[int(second)],
                )
                cell_blocks.append(_vertex_block(positions, normals, colour, lit=False))

        if self._display_scene.polyhedra:
            opaque_offset = 0
            transparent_offset = 0
            transparent_edge_offset = 0
            edge_lift = max(1.0e-4, 8.0e-4 * float(self._display_scene.base_radius))
            for poly_index in (range(len(self._display_scene.polyhedra)) if poly_indices is None else poly_indices):
                polyhedron = self._display_scene.polyhedra[poly_index]
                opaque_first = opaque_offset
                colour = self._polyhedron_colour(polyhedron)
                # Geometry always keeps its host/default material. Indexed
                # overrides live only in the table, so removing the last one
                # can disable the extra vertex fetch without rewriting meshes.
                alpha = self._polyhedron_opacity(poly_index, use_local=False)
                opaque = alpha == 1
                edge_colour = QColor(20, 20, 20) if self.engraving else _polyhedron_edge_colour(colour)
                poly_center = np.asarray(polyhedron.center, dtype=float)
                edge_faces: dict[tuple[int, int], list[np.ndarray]] = {}
                for face_index, face in enumerate(polyhedron.faces):
                    world, positions, normals = self._cached_poly_face_mesh(
                        poly_index, face_index, polyhedron, face
                    )
                    block = self._triangle_instances(positions, colour, alpha)
                    block[:,3] = poly_index + 1
                    if opaque:
                        opaque_blocks.append(block)
                        opaque_offset += block.shape[0]
                    else:
                        face_blocks.append(block)
                        face_ranges.append(_PolyhedronRange(
                            polyhedron.site_key, transparent_offset, block.shape[0], world.mean(axis=0), poly_index
                        ))
                        transparent_offset += block.shape[0]

                    # Collect one outward normal per adjacent face.  A shared edge
                    # is emitted only once below, lifted along the exterior
                    # dihedral bisector instead of radially from the polyhedron
                    # centre.  The radial lift fails for obtuse dihedrals because
                    # it can still leave the line inside one of the two faces.
                    face_normal = np.cross(world[1] - world[0], world[2] - world[0])
                    normal_length = float(np.linalg.norm(face_normal))
                    if normal_length > 1.0e-12:
                        face_normal /= normal_length
                        if float(np.dot(face_normal, world.mean(axis=0) - poly_center)) < 0.0:
                            face_normal = -face_normal
                        for first_index, second_index in zip(face, face[1:] + face[:1]):
                            edge_key = tuple(sorted((int(first_index), int(second_index))))
                            edge_faces.setdefault(edge_key, []).append(face_normal.copy())

                for (first_index, second_index), adjacent_normals in edge_faces.items():
                    first_point = np.asarray(polyhedron.vertices[first_index], dtype=float)
                    second_point = np.asarray(polyhedron.vertices[second_index], dtype=float)
                    outward = np.sum(np.asarray(adjacent_normals, dtype=float), axis=0)
                    outward_length = float(np.linalg.norm(outward))
                    if outward_length <= 1.0e-12:
                        # Degenerate / almost-flat fallback.
                        midpoint = 0.5 * (first_point + second_point)
                        outward = midpoint - poly_center
                        outward_length = float(np.linalg.norm(outward))
                    if outward_length > 1.0e-12:
                        outward = outward / outward_length * edge_lift
                    else:
                        outward = np.zeros(3, dtype=float)
                    edge_block = self._segment_instance(
                        first_point + outward, second_point + outward, edge_colour,
                        alpha=1.0 if self.engraving else alpha,
                    )
                    edge_block[:,3] = poly_index + 1
                    face_edge_blocks.append(edge_block)
                    midpoint = 0.5 * (first_point + second_point)
                    face_edge_ranges.append(_PolyhedronRange(
                        polyhedron.site_key, transparent_edge_offset, edge_block.shape[0], midpoint, poly_index
                    ))
                    transparent_edge_offset += edge_block.shape[0]
                if opaque_offset > opaque_first:
                    opaque_ranges.append(_PolyhedronRange(
                        polyhedron.site_key, opaque_first, opaque_offset - opaque_first, poly_index=poly_index
                    ))

        return (
            np.empty((0, _ATOM_INSTANCE_FLOATS), dtype=np.float32),
            self._combine_instances(opaque_blocks, _TRIANGLE_INSTANCE_FLOATS),
            _combine_blocks(cell_blocks),
            self._combine_instances(face_blocks, _TRIANGLE_INSTANCE_FLOATS),
            face_ranges,
            self._combine_instances(face_edge_blocks, _SEGMENT_INSTANCE_FLOATS),
            face_edge_ranges,
            opaque_ranges,
        )

    def _hatch_plan(self) -> list[tuple[int, int]]:
        """Keep the original density normalization over visible polyhedra."""
        if self._display_scene is None or not self.engraving or not self.hatching:
            return []
        selected = [
            (index, float((self.orientation @ polyhedron.center)[2]))
            for index, polyhedron in enumerate(self._display_scene.polyhedra)
            if self.polyhedron_site_visibility.get(polyhedron.site_key, True)
            and self._polyhedron_opacity(index) > 0
        ]
        if not selected:
            return []
        depths = [depth for _index, depth in selected]
        minimum, maximum = min(depths), max(depths)
        extent = maximum - minimum
        return [
            (index, hatch_division_count(
                0.5 if abs(extent) < 1e-12 else (maximum - depth) / extent,
                self.hatch_density, self.hatch_grip,
            ))
            for index, depth in selected
        ]

    def _prepare_hatch_view(self) -> tuple:
        key = (
            np.asarray(self.orientation, dtype=float).tobytes(),
            self.width(), self.height(), self.zoom, self.pan_x, self.pan_y,
        )
        if key != self._hatch_view_key:
            self._hatch_mesh_cache.clear()
            self._hatch_view_key = key
        return key

    def _cached_polyhedron_hatches(self, poly_index: int, division_count: int) -> np.ndarray:
        variants = self._hatch_mesh_cache.setdefault(poly_index, OrderedDict())
        if division_count not in variants:
            variants[division_count] = self._build_polyhedron_hatch_geometry(poly_index, division_count)
            if len(variants) > _HATCH_VARIANTS_PER_POLYHEDRON:
                variants.popitem(last=False)
        variants.move_to_end(division_count)
        return variants[division_count]

    def _build_hatch_geometry(self) -> np.ndarray:
        """CPU representation, also useful without an initialized GL context."""
        self._prepare_hatch_view()
        return _combine_blocks([
            self._cached_polyhedron_hatches(index, divisions)
            for index, divisions in self._hatch_plan()
        ])

    def _build_polyhedron_hatch_geometry(self, poly_index: int, division_count: int) -> np.ndarray:
        polyhedron = self._display_scene.polyhedra[poly_index]
        hatch_blocks: list[np.ndarray] = []
        inverse_orientation = np.linalg.inv(np.asarray(self.orientation, dtype=float))
        hatch_colour = QColor(30, 30, 30)
        hatch_lift = max(1.0e-4, 1.0e-3 * float(self._display_scene.base_radius))
        vertices_camera = np.asarray(
            [self.orientation @ np.asarray(vertex, dtype=float) for vertex in polyhedron.vertices],
            dtype=float,
        )
        projected = [self._project_camera_point(vertex) for vertex in vertices_camera]
        normals: dict[int, np.ndarray] = {}
        visible_faces: list[int] = []
        for face_index, face in enumerate(polyhedron.faces):
            face_vertices = vertices_camera[list(face)]
            normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
            length = float(np.linalg.norm(normal))
            if length < 1e-12:
                continue
            normal /= length
            normals[face_index] = normal
            if float(normal[2]) > 1e-9:
                visible_faces.append(face_index)
        if not visible_faces:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
        hatches = _polyhedron_hatches(
            polyhedron,
            vertices_camera,
            projected,
            visible_faces,
            normals,
            division_count,
            self._project_camera_point,
        )
        for face_index, segments in hatches.items():
            if not segments:
                continue
            face = polyhedron.faces[face_index]
            plane_point = vertices_camera[int(face[0])]
            plane_normal = normals[face_index]
            hatch_offset = plane_normal * hatch_lift
            edge_segments: list[np.ndarray] = []
            edge_normals: list[np.ndarray] = []
            for first, second in segments:
                first_camera = self._unproject_to_face(first, plane_point, plane_normal)
                second_camera = self._unproject_to_face(second, plane_point, plane_normal)
                if first_camera is None or second_camera is None:
                    continue
                first_world = inverse_orientation @ (first_camera + hatch_offset)
                second_world = inverse_orientation @ (second_camera + hatch_offset)
                positions, vertex_normals = line_vertices(first_world, second_world)
                edge_segments.append(positions)
                edge_normals.append(vertex_normals)
            if edge_segments:
                hatch_blocks.append(
                    _vertex_block(
                        np.concatenate(edge_segments, axis=0),
                        np.concatenate(edge_normals, axis=0),
                        hatch_colour,
                        alpha=1.0,
                        lit=False,
                    )
                )
        return _combine_blocks(hatch_blocks)

    def _build_engraving_bond_instances(self, bond_indices=None) -> np.ndarray:
        blocks = []
        for index in self._selected_bond_ids(bond_indices):
            first, second = (int(i) for i in self._display_scene.bonds[index])
            a, b = self._display_scene.atom_centers[[first, second]]
            if not self._locally_visible(first) or not self._locally_visible(second):
                b = a
            blocks.append(self._segment_instance(a, b, QColor('#777777')))
        return self._combine_instances(blocks, _SEGMENT_INSTANCE_FLOATS)

    @staticmethod
    def _atom_outline_mesh() -> np.ndarray:
        """Preserve the original 64 solid segments and 32 short BW dashes."""
        vertices = []
        for index in range(_OUTLINE_SEGMENTS):
            start = 2.0 * math.pi * index / _OUTLINE_SEGMENTS
            for endpoint in (0, 1):
                solid = 2.0 * math.pi * (index + endpoint) / _OUTLINE_SEGMENTS
                # Odd dashed segments collapse to a point; the existing edge
                # geometry shader rejects them before emitting any triangles.
                dashed = start if index % 2 else (
                    2.0 * math.pi * (index + 0.58 * endpoint) / _OUTLINE_SEGMENTS
                )
                vertices.append((math.cos(solid), math.sin(solid),
                                 math.cos(dashed), math.sin(dashed)))
        return np.asarray(vertices, dtype=np.float32)

    def _build_atom_outline_instances(self) -> np.ndarray:
        if self._display_scene is None or not self.show_atoms:
            return np.empty((0, _OUTLINE_INSTANCE_FLOATS), dtype=np.float32)
        records = []
        for atom_index, (atom, center) in enumerate(zip(self._display_scene.atoms, self._display_scene.atom_centers)):
            if not self._atom_visible(atom):
                continue
            if atom.external and not self.show_external_atoms:
                continue
            occupancy = sum(max(0.0, float(component.occupancy)) for component in atom.components)
            partial = self.engraving and occupancy < 1.0 - 1.0e-8
            colour = QColor(30, 30, 30) if self.engraving else _blend_components(atom.components).darker(165)
            records.append((*center, self._atom_radius(atom) if self._locally_visible(atom_index) else -1.0, colour.redF(),
                            colour.greenF(), colour.blueF(), colour.alphaF(), float(partial)))
        return np.asarray(records, dtype=np.float32).reshape(-1, _OUTLINE_INSTANCE_FLOATS)

    @staticmethod
    def _point_in_convex_face(point: np.ndarray, polygon: np.ndarray, normal: np.ndarray) -> bool:
        """Return True when *point* lies inside/on one convex planar polygon."""
        poly = np.asarray(polygon, dtype=float).reshape(-1, 3)
        p = np.asarray(point, dtype=float).reshape(3)
        n = np.asarray(normal, dtype=float).reshape(3)
        if len(poly) < 3:
            return False
        signs: list[float] = []
        scale = max(1.0, float(np.max(np.linalg.norm(poly - poly[0], axis=1))))
        tolerance = 1.0e-9 * scale
        for index, first in enumerate(poly):
            second = poly[(index + 1) % len(poly)]
            value = float(np.dot(np.cross(second - first, p - first), n))
            if abs(value) > tolerance:
                signs.append(value)
        if not signs:
            return True
        return all(value >= 0.0 for value in signs) or all(value <= 0.0 for value in signs)

    @staticmethod
    def _points_in_convex_face(points: np.ndarray, polygon: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """The same finite-face test for all section-segment midpoints at once."""
        poly = np.asarray(polygon, dtype=float).reshape(-1, 3)
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if len(poly) < 3:
            return np.zeros(len(points), dtype=bool)
        scale = max(1.0, float(np.max(np.linalg.norm(poly - poly[0], axis=1))))
        tolerance = 1.0e-9 * scale
        edges = np.roll(poly, -1, axis=0) - poly
        sides = np.cross(edges[None, :, :], points[:, None, :] - poly[None, :, :]) @ normal
        return np.all(sides >= -tolerance, axis=1) | np.all(sides <= tolerance, axis=1)

    def _build_atom_face_intersection_geometry(self, *, poly_indices=None) -> tuple[np.ndarray, list[_PolyhedronRange]]:
        """Build the actual sphere/face intersection curves.

        Each polyhedron face is a finite planar polygon.  A sphere/plane
        intersection is a circle; only the circle pieces that lie inside the
        finite face polygon are emitted.  Hidden parts are then rejected by the
        normal OpenGL depth test.
        """
        if self._display_scene is None:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32), []

        blocks: list[np.ndarray] = []
        ranges: list[_PolyhedronRange] = []
        offset = 0
        samples = 160
        angles = [2.0 * math.pi * index / samples for index in range(samples)]
        circle_cos = np.asarray([math.cos(angle) for angle in angles])[:, None]
        circle_sin = np.asarray([math.sin(angle) for angle in angles])[:, None]
        visible_atoms = [
            (
                atom,
                np.asarray(center, dtype=float),
                self._atom_radius(atom),
            )
            for atom_index, (atom, center) in enumerate(zip(self._display_scene.atoms, self._display_scene.atom_centers))
            if self._atom_visible(atom) and self._locally_visible(atom_index)
            and (self.show_external_atoms or not atom.external)
        ]
        if not visible_atoms:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32), []
        atom_centres = np.asarray([item[1] for item in visible_atoms], dtype=float)
        atom_radii = np.asarray([item[2] for item in visible_atoms], dtype=float)

        for poly_index in (range(len(self._display_scene.polyhedra)) if poly_indices is None else poly_indices):
            polyhedron = self._display_scene.polyhedra[poly_index]
            # Coloured contact curves match their own face, including host
            # colour overrides. Engraving keeps the existing black ink.
            colour = QColor(20, 20, 20) if self.engraving else self._polyhedron_colour(polyhedron)
            poly_first = offset
            vertices = np.asarray(polyhedron.vertices, dtype=float)
            poly_center = np.asarray(polyhedron.center, dtype=float)
            for face in polyhedron.faces:
                polygon = vertices[list(face)]
                if len(polygon) < 3:
                    continue
                normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
                normal_length = float(np.linalg.norm(normal))
                if normal_length <= 1.0e-12:
                    continue
                normal /= normal_length
                face_center = polygon.mean(axis=0)
                if float(np.dot(normal, face_center - poly_center)) < 0.0:
                    normal = -normal

                # Stable basis in the face plane for the section circle.
                u = polygon[1] - polygon[0]
                u -= float(np.dot(u, normal)) * normal
                u_length = float(np.linalg.norm(u))
                if u_length <= 1.0e-12:
                    continue
                u /= u_length
                v = np.cross(normal, u)
                v_length = float(np.linalg.norm(v))
                if v_length <= 1.0e-12:
                    continue
                v /= v_length

                # Cheap vectorised broad phase: almost every atom is far from
                # a given face in a real structure, so do not enter Python for
                # every atom×face pair.
                signed_all = (atom_centres - polygon[0]) @ normal
                bbox_min = polygon.min(axis=0)
                bbox_max = polygon.max(axis=0)
                in_box = np.all(atom_centres >= bbox_min - atom_radii[:, None], axis=1) & np.all(
                    atom_centres <= bbox_max + atom_radii[:, None], axis=1
                )
                candidate_indices = np.nonzero((np.abs(signed_all) < atom_radii - 1.0e-8) & in_box)[0]
                for atom_index in candidate_indices:
                    _atom, atom_center, atom_radius = visible_atoms[int(atom_index)]
                    signed_distance = float(signed_all[int(atom_index)])
                    circle_radius_sq = atom_radius * atom_radius - signed_distance * signed_distance
                    if circle_radius_sq <= 1.0e-12:
                        continue
                    circle_radius = math.sqrt(circle_radius_sq)
                    circle_center = atom_center - signed_distance * normal

                    points = circle_center + circle_radius * (circle_cos * u + circle_sin * v)
                    following = np.roll(points, -1, axis=0)
                    inside = self._points_in_convex_face(0.5 * (points + following), polygon, normal)
                    if np.any(inside):
                        positions = np.stack((points[inside], following[inside]), axis=1).reshape(-1, 3)
                        block = _vertex_block(
                            positions,
                            np.zeros_like(positions),
                            colour,
                            lit=False,
                        )
                        blocks.append(block)
                        offset += len(block)
            if offset > poly_first:
                ranges.append(_PolyhedronRange(polyhedron.site_key, poly_first, offset - poly_first, poly_index=poly_index))

        return _combine_blocks(blocks), ranges

    def _delete_primitive_extras(self, prefix: str) -> None:
        chunks = self._primitive_chunks.get(prefix, [])
        for vbo, texture, _count in chunks[1:]:
            glDeleteTextures(1, [texture])
            glDeleteBuffers(1, [vbo])
        self._primitive_chunks[prefix] = chunks[:1]

    def _upload_primitive(self, prefix: str, records: np.ndarray) -> int:
        """Split storage at the driver limit without changing global draw ranges."""
        self._delete_primitive_extras(prefix)
        capacity = max(1, self._primitive_texel_capacity // (records.shape[1] // 4))
        base_vbo, base_texture = getattr(self, prefix + '_vbo'), getattr(self, prefix + '_texture')
        chunks = []
        self._primitive_chunks[prefix] = chunks
        for first in range(0, max(1, len(records)), capacity):
            block = records[first:first + capacity]
            if first == 0:
                vbo, texture = base_vbo, base_texture
            else:
                vbo = int(glGenBuffers(1))
                chunks.append((vbo, 0, 0))
                texture = int(glGenTextures(1))
                chunks[-1] = (vbo, texture, len(block))
            if first == 0:
                chunks.append((vbo, texture, len(block)))
            self._upload_static(getattr(self, prefix + '_vao'), vbo, block)
            if first:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_BUFFER, texture)
                glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, vbo)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        return len(records)

    def _rebuild_gpu_cache(self) -> None:
        self._delete_partial_poly_buffers()
        _atoms, opaque, cell, faces, ranges, face_edges, face_edge_ranges, opaque_ranges = self._build_static_geometry()
        self._opaque_count = self._upload_primitive('_opaque', opaque)
        self._base_ranges_by_poly.clear()
        self._opaque_ranges = opaque_ranges
        self._cell_count = self._upload_static(self._cell_vao, self._cell_vbo, cell)
        self._face_count = self._upload_primitive('_face', faces)
        self._face_ranges = ranges
        self._face_edge_count = self._upload_primitive('_face_edge', face_edges)
        self._face_edge_ranges = face_edge_ranges
        self._face_edge_order_records = None
        self._classified_poly_ranges = None
        self._poly_opacity_data = self.get_polyhedron_opacities().astype(np.float32).reshape(-1,1)
        if self._poly_opacity_vbo:
            glBindBuffer(GL_ARRAY_BUFFER, self._poly_opacity_vbo)
            records = self._poly_opacity_data if len(self._poly_opacity_data) else np.ones((1,1),np.float32)
            glBufferData(GL_ARRAY_BUFFER, records.nbytes, records, GL_DYNAMIC_DRAW)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_BUFFER, self._poly_opacity_texture)
            glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self._poly_opacity_vbo)
            glBindTexture(GL_TEXTURE_BUFFER, 0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        self._pending_opacities.clear()
        self._gpu_dirty = False
        self._pending_polyhedra.clear()

    def _rebuild_bond_cache(self) -> None:
        self._bond_gpu_rows = {index: row for row, index in enumerate(self._selected_bond_ids())}
        colour_bonds = self._build_colour_bond_instances()
        self._bond_instance_data = colour_bonds
        self._bond_count = self._upload_static(self._bond_vao, self._bond_vbo, colour_bonds)
        engraving_bonds = self._build_engraving_bond_instances()
        self._engraving_instance_data = engraving_bonds
        self._engraving_bond_count = self._upload_primitive('_engraving_bond', engraving_bonds)
        self._bond_dirty = False
        self._pending_bonds.clear()

    def _rebuild_atom_cache(self) -> None:
        atoms = self._build_atom_instances()
        self._atom_instance_data = atoms
        self._atom_gpu_rows = {i: row for row, i in enumerate(
            i for i in range(len(self._display_scene.atoms)) if self._global_atom_visible(i))}
        self._atom_count = self._upload_static(self._atom_vao, self._atom_vbo, atoms)
        self._atom_dirty = False

    def _rebuild_intersection_cache(self) -> None:
        self._delete_partial_section_buffers()
        intersections, ranges = self._build_atom_face_intersection_geometry()
        self._intersection_count = self._upload_static(
            self._intersection_vao, self._intersection_vbo, intersections
        )
        self._intersection_ranges = ranges
        self._base_ranges_by_poly.pop('section', None)
        self._intersection_dirty = False
        self._pending_sections.clear()

    def _rebuild_hatch_cache(self) -> None:
        if not self.context() or not self.context().isValid():
            return
        view_key = self._prepare_hatch_view()
        plan = self._hatch_plan()
        for index, divisions in plan:
            buffer = self._hatch_buffers.get(index)
            if buffer is None:
                buffer = _HatchBuffer(*self._create_buffer())
                self._hatch_buffers[index] = buffer
            geometry_key = (view_key, divisions, self._poly_revisions.get(index, 0))
            if buffer.geometry_key != geometry_key:
                hatches = self._cached_polyhedron_hatches(index, divisions)
                buffer.count = self._upload_static(buffer.vao, buffer.vbo, hatches)
                buffer.geometry_key = geometry_key
        self._hatch_draw_indices = tuple(index for index, _divisions in plan)
        self._hatch_count = sum(self._hatch_buffers[index].count for index in self._hatch_draw_indices)
        self._hatch_dirty = False

    def _delete_hatch_buffers(self) -> None:
        """Called only while the widget's OpenGL context is current."""
        for buffer in self._hatch_buffers.values():
            glDeleteBuffers(1, [buffer.vbo])
            glDeleteVertexArrays(1, [buffer.vao])
        self._hatch_buffers.clear()
        self._hatch_draw_indices = ()
        self._hatch_count = 0
        self._hatch_buffers_need_clear = False

    def _rebuild_atom_outline_cache(self) -> None:
        if not self.context() or not self.context().isValid():
            return
        outlines = self._build_atom_outline_instances()
        self._outline_instance_data = outlines
        self._outline_gpu_rows = {i: row for row, i in enumerate(
            i for i in range(len(self._display_scene.atoms)) if self.show_atoms and self._global_atom_visible(i))}
        self._atom_outline_count = self._upload_static(
            self._atom_outline_vao, self._atom_outline_vbo, outlines
        )
        self._atom_outline_dirty = False

    @staticmethod
    def _row_runs(rows):
        rows = sorted(set(int(i) for i in rows))
        if not rows:
            return
        first = previous = rows[0]
        for index in rows[1:]:
            if index != previous + 1:
                yield first, previous + 1
                first = index
            previous = index
        yield first, previous + 1

    def _patch_rows(self, vbo, data, rows) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        for first, finish in self._row_runs(rows):
            block = np.ascontiguousarray(data[first:finish])
            glBufferSubData(GL_ARRAY_BUFFER, first * data.shape[1] * 4, block.nbytes, block)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _patch_primitive_rows(self, prefix, data, rows) -> None:
        rows = set(rows)
        offset = 0
        for vbo, _texture, size in self._primitive_chunks[prefix]:
            local = [i - offset for i in rows if offset <= i < offset + size]
            if local:
                self._patch_rows(vbo, data[offset:offset + size], local)
            offset += size

    def _upload_changed_primitive(self, prefix, records) -> int:
        chunks = self._primitive_chunks.get(prefix, [])
        capacity = max(1, self._primitive_texel_capacity // (records.shape[1] // 4))
        sizes = [len(records[i:i + capacity]) for i in range(0, max(1, len(records)), capacity)]
        if [item[2] for item in chunks] != sizes:
            return self._upload_primitive(prefix, records)
        offset = 0
        for vbo, _texture, size in chunks:
            if size:
                self._patch_rows(vbo, records[offset:offset + size], range(size))
            offset += size
        return len(records)

    def _delete_partial_poly_buffers(self) -> None:
        for data in self._partial_poly_gpu.values():
            for prefix in data['prefixes']:
                self._delete_primitive_extras(prefix)
                self._primitive_chunks.pop(prefix, None)
                for suffix in ('_texture', '_vbo', '_vao', '_order_vbo'):
                    value = getattr(self, prefix + suffix, 0)
                    if value:
                        if suffix == '_texture':
                            glDeleteTextures(1, [value])
                        elif suffix == '_vao':
                            glDeleteVertexArrays(1, [value])
                        else:
                            glDeleteBuffers(1, [value])
                    if hasattr(self, prefix + suffix):
                        delattr(self, prefix + suffix)
                for suffix in ('_order_records', '_order_capacity'):
                    if hasattr(self, prefix + suffix):
                        delattr(self, prefix + suffix)
        self._partial_poly_gpu.clear()

    def _delete_partial_section_buffers(self) -> None:
        for buffer in self._partial_section_gpu.values():
            glDeleteBuffers(1, [buffer.vbo])
            glDeleteVertexArrays(1, [buffer.vao])
        self._partial_section_gpu.clear()

    def _update_one_polyhedron_gpu(self, index) -> None:
        self._classified_poly_ranges = None
        geometry = self._build_static_geometry(poly_indices=(index,), include_cell=False)
        data = self._partial_poly_gpu.setdefault(index, {'prefixes': []})
        for kind, array, ranges in (('opaque', geometry[1], geometry[7]),
                                    ('face', geometry[3], geometry[4]),
                                    ('face_edge', geometry[5], geometry[6])):
            prefix = f'_partial_poly_{index}_{kind}'
            if prefix not in data['prefixes']:
                data['prefixes'].append(prefix)
                self._create_primitive_buffer(prefix)
            self._upload_changed_primitive(prefix, array)
            data[kind] = [replace(item, buffer_key=prefix) for item in ranges]
            if kind == 'face_edge':
                setattr(self, prefix + '_order_records', None)

    def _update_one_section_gpu(self, index) -> None:
        records, _ranges = self._build_atom_face_intersection_geometry(poly_indices=(index,))
        buffer = self._partial_section_gpu.get(index)
        if buffer is None:
            buffer = _HatchBuffer(*self._create_buffer())
            self._partial_section_gpu[index] = buffer
        if buffer.count == len(records):
            self._patch_rows(buffer.vbo, records, range(len(records)))
        else:
            buffer.count = self._upload_static(buffer.vao, buffer.vbo, records)

    def _stored_ranges(self, kind) -> list[_PolyhedronRange]:
        base = getattr(self, {'opaque': '_opaque_ranges', 'face': '_face_ranges',
                              'face_edge': '_face_edge_ranges', 'section': '_intersection_ranges'}[kind])
        overrides = self._partial_section_gpu if kind == 'section' else self._partial_poly_gpu
        if not overrides:
            return base
        grouped = self._base_ranges_by_poly.get(kind)
        if grouped is None:
            grouped = {}
            for item in base:
                grouped.setdefault(item.poly_index, []).append(item)
            self._base_ranges_by_poly[kind] = grouped
        result = []
        for index, poly in enumerate(self._display_scene.polyhedra):
            if index not in overrides:
                result.extend(grouped.get(index, []))
            elif kind == 'section':
                buffer = overrides[index]
                if buffer.count:
                    result.append(_PolyhedronRange(poly.site_key, 0, buffer.count,
                                                   poly_index=index, buffer_key=f'section:{index}'))
            else:
                result.extend(overrides[index].get(kind, []))
        return result

    def _effective_ranges(self, kind) -> list[_PolyhedronRange]:
        if kind not in ('opaque','face'):
            return self._stored_ranges(kind)
        if self._classified_poly_ranges is None:
            classified = {'opaque': [], 'face': []}
            for source in ('opaque','face'):
                for item in self._stored_ranges(source):
                    opacity = self._polyhedron_opacity(item.poly_index)
                    if opacity <= 0:
                        continue
                    target = 'opaque' if opacity == 1 else 'face'
                    if source != target and not item.buffer_key:
                        item = replace(item, buffer_key='_opaque' if source == 'opaque' else '_face')
                    classified[target].append(item)
            self._classified_poly_ranges = classified
        return self._classified_poly_ranges[kind]

    def _flush_scene_updates(self) -> None:
        opacity_rows = sorted(self._pending_opacities)
        for index in opacity_rows:
            if index < len(self._poly_opacity_data):
                self._poly_opacity_data[index,0] = self._polyhedron_opacity(index)
        if opacity_rows and self._poly_opacity_vbo:
            self._patch_rows(self._poly_opacity_vbo, self._poly_opacity_data, opacity_rows)
        self._pending_opacities.clear()
        for mapping, data, vbo in ((self._atom_gpu_rows, self._atom_instance_data, self._atom_vbo),
                                    (self._outline_gpu_rows, self._outline_instance_data, self._atom_outline_vbo)):
            rows = []
            for index in sorted(self._pending_atoms):
                row = mapping.get(index)
                if row is None:
                    continue
                data[row, :3] = self._display_scene.atom_centers[index]
                data[row, 3] = self._atom_radius(self._display_scene.atoms[index]) if self._locally_visible(index) else -1.0
                rows.append(row)
            if rows:
                self._patch_rows(vbo, data, rows)
        self._pending_atoms.clear()
        ids = [i for i in sorted(self._pending_bonds) if i in self._bond_gpu_rows]
        if ids:
            colour = self._build_colour_bond_instances(bond_indices=ids)
            engraving = self._build_engraving_bond_instances(bond_indices=ids)
            colour_rows = []
            engraving_rows = []
            for source, index in enumerate(ids):
                row = self._bond_gpu_rows[index]
                self._bond_instance_data[2*row:2*row+2] = colour[2*source:2*source+2]
                self._engraving_instance_data[row] = engraving[source]
                colour_rows.extend((2*row, 2*row+1)); engraving_rows.append(row)
            self._patch_rows(self._bond_vbo, self._bond_instance_data, colour_rows)
            self._patch_primitive_rows('_engraving_bond', self._engraving_instance_data, engraving_rows)
        self._pending_bonds.clear()
        for index in sorted(self._pending_polyhedra):
            self._update_one_polyhedron_gpu(index)
        self._pending_polyhedra.clear()
        if self.show_atoms and self.show_polyhedra:
            for index in sorted(self._pending_sections):
                self._update_one_section_gpu(index)
            self._pending_sections.clear()

    def _set_frame_uniforms(self) -> None:
        glUseProgram(self._program)
        glUniformMatrix4fv(self._uniform_projection, 1, GL_TRUE, self._projection_matrix())
        glUniformMatrix3fv(
            self._uniform_orientation,
            1,
            GL_TRUE,
            np.asarray(self.orientation, dtype=np.float32),
        )
        glUniform3f(self._uniform_light, -0.35, 0.45, 0.82)

    def _set_atom_uniforms(self) -> None:
        glUseProgram(self._atom_program)
        glUniformMatrix4fv(self._atom_uniform_projection, 1, GL_TRUE, self._projection_matrix())
        glUniformMatrix3fv(
            self._atom_uniform_orientation,
            1,
            GL_TRUE,
            np.asarray(self.orientation, dtype=np.float32),
        )
        glUniform3f(self._atom_uniform_light, -0.35, 0.45, 0.82)
        glUniform1f(self._atom_uniform_engraving, 1.0 if self.engraving else 0.0)

    def _set_edge_uniforms(self, *, width_px: float = 1.575, depth_bias: float = 1.0e-5) -> None:
        glUseProgram(self._edge_program)
        glUniformMatrix4fv(self._edge_uniform_projection, 1, GL_TRUE, self._projection_matrix())
        glUniformMatrix3fv(
            self._edge_uniform_orientation,
            1,
            GL_TRUE,
            np.asarray(self.orientation, dtype=np.float32),
        )
        device_ratio = max(1.0, float(self.devicePixelRatioF())) * self._render_scale
        glUniform2f(
            self._edge_uniform_viewport,
            max(1.0, float(self.width()) * device_ratio),
            max(1.0, float(self.height()) * device_ratio),
        )
        glUniform1f(self._edge_uniform_width, float(width_px) * device_ratio)
        glUniform1f(self._edge_uniform_depth_bias, float(depth_bias))
        glUniform1f(self._edge_uniform_opacity, 1.0)

    def _set_instance_uniforms(self, program: int, record_texels: int = 0,
                               *, width_px: float = 1.575, depth_bias: float = 1e-5) -> None:
        uniforms = self._instance_uniforms[program]
        glUseProgram(program)
        glUniformMatrix4fv(uniforms['u_projection'], 1, GL_TRUE, self._projection_matrix())
        glUniformMatrix3fv(uniforms['u_orientation'], 1, GL_TRUE,
                           np.asarray(self.orientation, dtype=np.float32))
        if program == self._bond_program:
            glUniform3f(uniforms['u_light_dir'], -0.35, 0.45, 0.82)
        else:
            glUniform1i(uniforms['u_use_order'], 0)
            glUniform1i(uniforms['u_records'], 0)
            glUniform1i(uniforms['u_record_texels'], record_texels)
            glUniform1i(uniforms['u_poly_opacities'], 3)
            enabled = bool(self._scene_updates and self._scene_updates.opacity_overrides)
            glUniform1i(uniforms['u_poly_opacities_enabled'], int(enabled))
            if enabled:
                glActiveTexture(GL_TEXTURE0 + 3)
                glBindTexture(GL_TEXTURE_BUFFER, self._poly_opacity_texture)
                glActiveTexture(GL_TEXTURE0)
        if program == self._segment_program or (self._depth_peeler and program == self._depth_peeler.edge_program):
            ratio = max(1., float(self.devicePixelRatioF())) * self._render_scale
            glUniform2f(uniforms['u_viewport'], max(1., self.width() * ratio),
                        max(1., self.height() * ratio))
            glUniform1f(uniforms['u_edge_width'], width_px * ratio)
            glUniform1f(uniforms['u_depth_bias'], depth_bias)

    def _bind_primitive(self, vao: int, texture: int) -> None:
        glBindVertexArray(vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, texture)
        self._bound_instance_texture = texture
        self._bound_primitive_chunks = next(
            (chunks for chunks in self._primitive_chunks.values() if chunks and chunks[0][1] == texture), []
        )

    def _draw_primitive_instances(self, mode: int, first: int, count: int, *, program=None) -> None:
        program = program or (self._triangle_program if mode == GL_TRIANGLES else self._segment_program)
        remaining = count
        for _vbo, texture, size in self._bound_primitive_chunks:
            if first >= size:
                first -= size
                continue
            drawn = min(remaining, size - first)
            if texture != self._bound_instance_texture:
                glBindTexture(GL_TEXTURE_BUFFER, texture)
                self._bound_instance_texture = texture
            glUniform1i(self._instance_uniforms[program]['u_first'], first)
            glDrawArraysInstanced(mode, 0, 3 if mode == GL_TRIANGLES else 2, drawn)
            remaining -= drawn
            if remaining <= 0:
                break
            first = 0

    def _draw_ordered_edge_instances(self, ordered: list[_PolyhedronRange], prefix='_face_edge') -> None:
        """Keep coordinates resident; upload only tiny sorted instance indices."""
        if not ordered:
            return
        indices = np.asarray([index for item in ordered for index in range(item.first, item.first + item.count)],
                             dtype=np.int32)
        glBindBuffer(GL_ARRAY_BUFFER, getattr(self, prefix + '_order_vbo'))
        if len(indices) > getattr(self, prefix + '_order_capacity'):
            glBufferData(GL_ARRAY_BUFFER, indices.nbytes, None, GL_DYNAMIC_DRAW)
            setattr(self, prefix + '_order_capacity', len(indices))
            setattr(self, prefix + '_order_records', None)
        if getattr(self, prefix + '_order_records') is None or not np.array_equal(indices, getattr(self, prefix + '_order_records')):
            glBufferSubData(GL_ARRAY_BUFFER, 0, indices.nbytes, indices)
            setattr(self, prefix + '_order_records', indices)
        uniforms = self._instance_uniforms[self._segment_program]
        glUniform1i(uniforms['u_use_order'], 1)
        chunks = self._bound_primitive_chunks
        capacity = chunks[0][2]
        # Runs retain the exact sorted edge order, even when storage is split
        # across texture buffers. The common single-buffer case is one draw.
        first = 0
        while first < len(indices):
            chunk_index = int(indices[first]) // capacity
            finish = first + 1
            while finish < len(indices) and int(indices[finish]) // capacity == chunk_index:
                finish += 1
            texture = chunks[chunk_index][1]
            if texture != self._bound_instance_texture:
                glBindTexture(GL_TEXTURE_BUFFER, texture)
                self._bound_instance_texture = texture
            glUniform1i(uniforms['u_first'], chunk_index * capacity)
            glVertexAttribIPointer(1, 1, GL_INT, 4, ctypes.c_void_p(first * 4))
            glDrawArraysInstanced(GL_LINES, 0, 2, finish - first)
            first = finish
        glUniform1i(uniforms['u_use_order'], 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def _draw_colour_bonds(self) -> None:
        if not self.show_bonds or self.engraving or self._bond_count <= 0:
            return
        self._set_instance_uniforms(self._bond_program)
        glBindVertexArray(self._bond_vao)
        glDrawElementsInstanced(GL_TRIANGLES, self._bond_index_count, GL_UNSIGNED_INT,
                               ctypes.c_void_p(0), self._bond_count)
        glBindVertexArray(0)

    @staticmethod
    def _draw_buffer(vao: int, count: int, mode: int) -> None:
        if count <= 0:
            return
        glBindVertexArray(vao)
        glDrawArrays(mode, 0, count)
        glBindVertexArray(0)

    def _visible_polyhedron_ranges(self, ranges: list[_PolyhedronRange]) -> list[_PolyhedronRange]:
        return [item for item in ranges if self.polyhedron_site_visibility.get(item.site_key, True)
                and (item.poly_index is None or self._polyhedron_opacity(item.poly_index) > 0)]

    def _draw_opaque_polyhedra(self) -> None:
        if not self.show_polyhedra:
            return
        ranges = self._visible_polyhedron_ranges(self._effective_ranges('opaque'))
        if not ranges:
            return
        self._set_instance_uniforms(self._triangle_program, 4)
        bound = None
        for item in ranges:
            prefix = item.buffer_key or '_opaque'
            if prefix != bound:
                self._bind_primitive(getattr(self, prefix + '_vao'), getattr(self, prefix + '_texture'))
                bound = prefix
            self._draw_primitive_instances(GL_TRIANGLES, item.first, item.count)
        glBindVertexArray(0)

    def _ordered_face_ranges(self) -> list[_PolyhedronRange]:
        orientation = np.asarray(self.orientation, dtype=float)
        return sorted(self._visible_polyhedron_ranges(self._effective_ranges('face')),
                      key=lambda item: float((orientation @ (item.center if item.center is not None
                                                            else self._display_scene.polyhedra[item.poly_index].center))[2]))

    def _draw_transparent_faces(self) -> None:
        if not self.show_polyhedra:
            return
        ranges = self._ordered_face_ranges()
        if not ranges:
            return
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        self._set_instance_uniforms(self._triangle_program, 4)
        bound = None
        for item in ranges:
            prefix = item.buffer_key or '_face'
            if prefix != bound:
                self._bind_primitive(getattr(self, prefix + '_vao'), getattr(self, prefix + '_texture'))
                bound = prefix
            self._draw_primitive_instances(GL_TRIANGLES, item.first, item.count)
        glBindVertexArray(0); glDepthMask(GL_TRUE); glDisable(GL_BLEND)

    def _draw_face_edges(self) -> None:
        if not self.show_polyhedra:
            return
        orientation = np.asarray(self.orientation, dtype=float)
        ordered = sorted(self._visible_polyhedron_ranges(self._effective_ranges('face_edge')),
                         key=lambda item: float((orientation @ item.center)[2]))
        if not ordered:
            return
        self._set_instance_uniforms(self._segment_program, 3, width_px=1.575, depth_bias=1e-5)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        first = 0
        while first < len(ordered):
            prefix = ordered[first].buffer_key or '_face_edge'
            finish = first + 1
            while finish < len(ordered) and (ordered[finish].buffer_key or '_face_edge') == prefix:
                finish += 1
            self._bind_primitive(getattr(self, prefix + '_vao'), getattr(self, prefix + '_texture'))
            self._draw_ordered_edge_instances(ordered[first:finish], prefix)
            first = finish
        glBindVertexArray(0); glDepthMask(GL_TRUE); glDisable(GL_BLEND)

    def _draw_hatches(self) -> None:
        if not self.show_polyhedra or not self.engraving or not self.hatching or self._hatch_count <= 0:
            return
        self._set_edge_uniforms(width_px=0.75, depth_bias=1.5e-5)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        for index in self._hatch_draw_indices:
            buffer = self._hatch_buffers[index]
            if buffer.count:
                glBindVertexArray(buffer.vao)
                glDrawArrays(GL_LINES, 0, buffer.count)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _draw_engraving_bonds(self) -> None:
        if not self.show_bonds or not self.engraving or self._engraving_bond_count <= 0:
            return
        self._set_instance_uniforms(self._segment_program, 3, width_px=1.8, depth_bias=0.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        self._bind_primitive(self._engraving_bond_vao, self._engraving_bond_texture)
        self._draw_primitive_instances(GL_LINES, 0, self._engraving_bond_count)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _draw_atom_outlines(self) -> None:
        if not self.show_atoms or self._atom_outline_count <= 0:
            return
        width_px = 0.9 if self.engraving else 0.95
        bias = 2.0e-5 if self.engraving else 1.4e-5
        uniforms = self._atom_outline_uniforms
        glUseProgram(self._atom_outline_program)
        glUniformMatrix4fv(uniforms["u_projection"], 1, GL_TRUE, self._projection_matrix())
        glUniformMatrix3fv(uniforms["u_orientation"], 1, GL_TRUE,
                           np.asarray(self.orientation, dtype=np.float32))
        glUniform1f(uniforms["u_outline_padding"], 0.45 / max(self._screen_scale(), 1e-12))
        device_ratio = max(1.0, float(self.devicePixelRatioF())) * self._render_scale
        glUniform2f(uniforms["u_viewport"], max(1.0, self.width() * device_ratio),
                    max(1.0, self.height() * device_ratio))
        glUniform1f(uniforms["u_edge_width"], width_px * device_ratio)
        glUniform1f(uniforms["u_depth_bias"], bias)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._atom_outline_vao)
        glDrawArraysInstanced(GL_LINES, 0, 2 * _OUTLINE_SEGMENTS, self._atom_outline_count)
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _draw_atom_face_intersections(self) -> None:
        if not self.show_atoms or not self.show_polyhedra:
            return
        ranges = self._visible_polyhedron_ranges(self._effective_ranges('section'))
        if not ranges:
            return
        width_px = 1.0 if self.engraving else 0.95
        bias = 7.5e-6 if self.engraving else 8.5e-6
        self._set_edge_uniforms(width_px=width_px, depth_bias=bias)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
        for item in ranges:
            local_opacity = (self._scene_updates.opacity_overrides.get(item.poly_index)
                             if self._scene_updates else None)
            glUniform1f(self._edge_uniform_opacity, self._polyhedron_opacity(item.poly_index)
                        if local_opacity is not None else 1.0)
            vao = (self._partial_section_gpu[item.poly_index].vao if item.buffer_key
                   else self._intersection_vao)
            glBindVertexArray(vao)
            glDrawArrays(GL_LINES, item.first, item.count)
        glBindVertexArray(0); glDepthMask(GL_TRUE); glDisable(GL_BLEND)

    def _foreground_colour(self) -> QColor:
        return QColor(25, 25, 25) if self.engraving else self.palette().text().color()

    def _current_fps(self) -> float | None:
        if len(self._frame_times) < 3:
            return None
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 1e-9:
            return None
        return (len(self._frame_times) - 1) / elapsed

    def _draw_overlay_cell(self, painter: QPainter) -> None:
        if self._display_scene is None or not self.show_cell:
            return
        pen = QPen(QColor(75, 75, 75))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern(list(_CELL_DASH_PATTERN))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for first, second in self._display_scene.cell_edges:
            painter.drawLine(
                self._project_world_point(self._display_scene.cell_vertices[int(first)]),
                self._project_world_point(self._display_scene.cell_vertices[int(second)]),
            )

    def _draw_overlay_basis(self, painter: QPainter) -> None:
        if self._display_scene is None or not self.show_basis:
            return
        directions = self.orientation @ (
            self._display_scene.basis_vectors / np.linalg.norm(self._display_scene.basis_vectors, axis=0)
        )
        origin = QPointF(56.0, self.height() - 52.0)
        colours = (QColor("#db2828"), QColor("#25a43a"), QColor("#2155dc"))
        for index, (name, colour) in enumerate(zip("abc", colours)):
            x, y = float(directions[0, index]), float(directions[1, index])
            length = math.hypot(x, y)
            pen = QPen(colour)
            pen.setWidthF(2.4)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QBrush(colour))
            if length <= 0.04:
                painter.drawEllipse(QRectF(origin.x() - 3.0, origin.y() - 3.0, 6.0, 6.0))
                painter.drawText(QPointF(origin.x() + 7.0, origin.y() - 7.0), name)
                continue
            target = QPointF(origin.x() + 43.0 * x, origin.y() - 43.0 * y)
            painter.drawLine(origin, target)
            vector = np.array([target.x() - origin.x(), target.y() - origin.y()], dtype=float)
            vector /= np.linalg.norm(vector)
            normal = np.array([-vector[1], vector[0]], dtype=float)
            tip = np.array([target.x(), target.y()], dtype=float)
            left = tip - 9.0 * vector + 4.0 * normal
            right = tip - 9.0 * vector - 4.0 * normal
            painter.drawPolygon(QPolygonF([target, QPointF(*left), QPointF(*right)]))
            painter.drawText(QPointF(target.x() + 7.0 * x, target.y() - 7.0 * y), name)
        painter.setBrush(QBrush(QColor("#777777")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(origin.x() - 2.5, origin.y() - 2.5, 5.0, 5.0))

    def _draw_overlay_legend(self, painter: QPainter) -> None:
        if self._display_scene is None or not self.show_atoms:
            return
        x, y = 12.0, 18.0
        metrics = QFontMetrics(painter.font())
        visible_elements = {
            component.element
            for atom in self._display_scene.atoms
            for component in atom.components
            if self._component_visible(component)
        }
        for element in self._display_scene.elements:
            if element not in visible_elements:
                continue
            component = next(
                component
                for atom in self._display_scene.atoms
                for component in atom.components
                if component.element == element
            )
            colour = self._component_colour(component)
            painter.setPen(QPen(colour.darker(150)))
            painter.setBrush(QBrush(colour))
            painter.drawEllipse(QRectF(x, y - 8.0, 10.0, 10.0))
            painter.setPen(QPen(self._foreground_colour()))
            painter.drawText(QPointF(x + 15.0, y), element)
            x += 22.0 + metrics.horizontalAdvance(element)
            if x > self.width() - 80.0:
                x = 12.0
                y += metrics.height() + 4.0

    def _draw_overlay(self) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(self._foreground_colour()))
            if self.opengl_error:
                painter.drawText(
                    self.rect().adjusted(20, 20, -20, -20),
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                    "OpenGL initialization failed:\n" + self.opengl_error,
                )
                return
            if self._display_scene is None:
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
                return
            if self.title_text:
                painter.drawText(
                    QRectF(8.0, 7.0, self.width() - 16.0, 24.0),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    self.title_text,
                )
            if self._show_fps:
                fps = self._current_fps()
                label = "FPS: …" if fps is None else f"FPS: {fps:.0f}"
                painter.drawText(QPointF(10.0, self.height() - 12.0), label)
            self._draw_overlay_cell(painter)
            self._draw_overlay_basis(painter)
            self._draw_overlay_legend(painter)
        finally:
            painter.end()

    def paintGL(self) -> None:
        if self._show_fps:
            self._frame_times.append(time.perf_counter())
        background = QColor(250, 250, 247) if self.engraving else self.palette().base().color()
        self._restore_render_state()
        glClearColor(background.redF(), background.greenF(), background.blueF(), 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.opengl_error is None and self._program and self._hatch_buffers_need_clear:
            self._delete_hatch_buffers()
            self._delete_partial_poly_buffers()
            self._delete_partial_section_buffers()
        if self.opengl_error is None and self._display_scene is not None and self._program:
            if self._gpu_dirty:
                self._rebuild_gpu_cache()
            if self._atom_dirty:
                self._rebuild_atom_cache()
            if self._bond_dirty:
                self._rebuild_bond_cache()
            if self._intersection_dirty and self.show_atoms and self.show_polyhedra:
                self._rebuild_intersection_cache()
            if self._hatch_dirty and self.show_polyhedra and self.engraving and self.hatching:
                self._rebuild_hatch_cache()
            if self._atom_outline_dirty:
                self._rebuild_atom_outline_cache()
            self._flush_scene_updates()
            peeled_faces = self._visible_polyhedron_ranges(self._effective_ranges('face')) if self.show_polyhedra and not self.engraving else []
            peeling = bool(peeled_faces)
            if peeling:
                try:
                    if self._depth_peeler is None:
                        self._depth_peeler = DepthPeeler(
                            self, _create_program, _PRIMITIVE_VERTEX_SHADER, _FRAGMENT_SHADER,
                            _EDGE_FRAGMENT_SHADER, _EDGE_GEOMETRY_SHADER,
                        )
                    ratio = max(1., float(self.devicePixelRatioF()))
                    self._depth_peeler.begin(max(1, round(self.width() * ratio)),
                                             max(1, round(self.height() * ratio)), background)
                except Exception as exc:
                    self.opengl_error = str(exc)
                    self._draw_overlay()
                    return
            elif self._depth_peeler:
                self._depth_peeler.release_targets()
            self._set_frame_uniforms()
            self._draw_opaque_polyhedra()
            self._draw_colour_bonds()
            if self.show_atoms and self._atom_count > 0:
                self._set_atom_uniforms()
                glBindVertexArray(self._atom_vao)
                glDrawElementsInstanced(
                    GL_TRIANGLES, self._atom_index_count, GL_UNSIGNED_INT,
                    ctypes.c_void_p(0), self._atom_count,
                )
                glBindVertexArray(0)
                self._set_frame_uniforms()

            # Atom silhouette and atom/face intersection ink belong to the atom
            # layer.  Draw them BEFORE transparent polyhedron faces so a face
            # that is actually in front blends over the ink instead of the ink
            # looking like a screen-space overlay.  Opaque geometry has already
            # populated the depth buffer, so the usual occlusion still applies.
            glUseProgram(0)
            self._draw_atom_face_intersections()
            self._draw_atom_outlines()

            self._set_frame_uniforms()
            if peeling:
                try:
                    self._depth_peeler.finish(peeled_faces,
                        self._visible_polyhedron_ranges(self._effective_ranges('face_edge')))
                except Exception as exc:
                    self._depth_peeler.restore()
                    self._depth_peeler._unbind_textures()
                    self._restore_render_state()
                    self.opengl_error = str(exc)
            else:
                self._draw_transparent_faces()
            glUseProgram(0)
            self._draw_engraving_bonds()
            self._draw_hatches()
            if not peeling:
                self._draw_face_edges()
            glUseProgram(0)
        # Leave the newly introduced texture-buffer target unbound for Qt's
        # painter/compositor and other widgets sharing this context group.
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        glActiveTexture(GL_TEXTURE0 + 3)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        glActiveTexture(GL_TEXTURE0)
        if self.scene is None and self._depth_peeler:
            self._depth_peeler.release_targets()
        self._draw_overlay()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
            and self._display_scene is not None
        ):
            self._last_mouse = event.position()
            self._drag_button = event.button()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        dx = position.x() - self._last_mouse.x()
        dy = position.y() - self._last_mouse.y()
        self._last_mouse = position
        if self._drag_button == Qt.MouseButton.LeftButton:
            if self.rotation_enabled:
                self.orientation = screen_drag_orientation(self.orientation, float(dx), float(dy))
                # Hatching is view-dependent: visible faces, seed face and propagated
                # directions must be recalculated continuously while the crystal turns.
                self._hatch_dirty = True
                self.orientation_changed.emit(self.orientation.copy())
                self.update()
        elif self._drag_button == Qt.MouseButton.RightButton:
            self.pan_x += float(dx)
            self.pan_y += float(dy)
            self._hatch_dirty = True
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == self._drag_button and self._last_mouse is not None:
            rotated = self._drag_button == Qt.MouseButton.LeftButton
            self._last_mouse = None
            self._drag_button = None
            if rotated and self.rotation_enabled:
                self.interaction_finished.emit()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta:
            self.zoom_by(delta / 120.0)
            event.accept()
            return
        super().wheelEvent(event)

    def _disconnect_context_cleanup(self) -> None:
        context = self._cleanup_context
        self._cleanup_context = None
        if context is None:
            return
        try:
            context.aboutToBeDestroyed.disconnect(self._context_about_to_be_destroyed)
        except (RuntimeError, TypeError):
            pass

    def _reset_gpu_objects(self) -> None:
        self._poly_opacity_vbo = 0
        self._poly_opacity_texture = 0
        self._poly_opacity_data = np.empty((0,1), np.float32)
        self._classified_poly_ranges = None
        self._pending_opacities.clear()
        self._depth_peeler = None
        self._render_scale = 1
        for attribute in (
            "_atom_vbo", "_atom_mesh_vbo", "_atom_ebo", "_opaque_vbo", "_bond_vbo", "_bond_mesh_vbo", "_bond_ebo", "_primitive_mesh_vbo", "_cell_vbo",
            "_face_vbo", "_face_edge_vbo", "_face_edge_order_vbo",
            "_atom_outline_vbo", "_atom_outline_mesh_vbo", "_intersection_vbo", "_engraving_bond_vbo",
            "_atom_vao", "_opaque_vao", "_bond_vao", "_cell_vao",
            "_face_vao", "_face_edge_vao",
            "_atom_outline_vao", "_intersection_vao", "_engraving_bond_vao",
            "_program", "_atom_program", "_edge_program", "_atom_outline_program", "_bond_program", "_triangle_program", "_segment_program",
        ):
            setattr(self, attribute, 0)
        for attribute in (
            "_uniform_projection", "_uniform_orientation", "_uniform_light",
            "_atom_uniform_projection", "_atom_uniform_orientation",
            "_atom_uniform_light", "_atom_uniform_engraving",
            "_edge_uniform_projection", "_edge_uniform_orientation",
            "_edge_uniform_viewport", "_edge_uniform_width",
            "_edge_uniform_depth_bias",
            "_edge_uniform_opacity",
        ):
            setattr(self, attribute, -1)
        for attribute in (
            "_bond_index_count", "_atom_count", "_atom_index_count", "_opaque_count", "_bond_count", "_cell_count",
            "_face_count", "_face_edge_count", "_hatch_count",
            "_atom_outline_count", "_intersection_count",
            "_engraving_bond_count",
        ):
            setattr(self, attribute, 0)
        self._base_ranges_by_poly = {}
        self._partial_poly_gpu = {}
        self._partial_section_gpu = {}
        self._atom_gpu_rows = {}; self._outline_gpu_rows = {}; self._bond_gpu_rows = {}
        self._atom_instance_data = np.empty((0, _ATOM_INSTANCE_FLOATS), np.float32)
        self._outline_instance_data = np.empty((0, _OUTLINE_INSTANCE_FLOATS), np.float32)
        self._bond_instance_data = np.empty((0, _BOND_INSTANCE_FLOATS), np.float32)
        self._engraving_instance_data = np.empty((0, _SEGMENT_INSTANCE_FLOATS), np.float32)
        self._atom_outline_uniforms = {}
        self._instance_uniforms = {}
        self._primitive_chunks = {}
        self._bound_primitive_chunks = []
        self._bound_instance_texture = 0
        self._face_edge_order_capacity = 0
        self._face_edge_order_records = None
        for attribute in ('_opaque_texture', '_face_texture', '_face_edge_texture', '_engraving_bond_texture'):
            setattr(self, attribute, 0)
        self._face_ranges = []
        self._face_edge_ranges = []
        self._opaque_ranges = []
        self._intersection_ranges = []
        self._hatch_buffers = {}
        self._hatch_draw_indices = ()
        self._hatch_buffers_need_clear = False
        self._gpu_dirty = True
        self._atom_dirty = True
        self._bond_dirty = True
        self._intersection_dirty = True
        self._hatch_dirty = True
        self._atom_outline_dirty = True

    def _context_about_to_be_destroyed(self) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self._cleaning_up:
            return
        self._cleaning_up = True
        context = self.context()
        can_delete = context is not None and context.isValid()
        if can_delete:
            self.makeCurrent()
        try:
            if can_delete:
                if self._poly_opacity_texture:
                    glDeleteTextures(1, [self._poly_opacity_texture])
                if self._poly_opacity_vbo:
                    glDeleteBuffers(1, [self._poly_opacity_vbo])
                if self._depth_peeler:
                    self._depth_peeler.close()
                self._delete_partial_poly_buffers()
                self._delete_partial_section_buffers()
                self._delete_hatch_buffers()
                for prefix in tuple(self._primitive_chunks):
                    self._delete_primitive_extras(prefix)
                for attribute in ('_opaque_texture', '_face_texture', '_face_edge_texture', '_engraving_bond_texture'):
                    value = getattr(self, attribute)
                    if value:
                        glDeleteTextures(1, [value])
                for attribute in (
                    "_atom_vbo", "_atom_mesh_vbo", "_atom_ebo", "_opaque_vbo", "_bond_vbo", "_bond_mesh_vbo", "_bond_ebo", "_primitive_mesh_vbo", "_cell_vbo",
                    "_face_vbo", "_face_edge_vbo", "_face_edge_order_vbo",
                    "_atom_outline_vbo", "_atom_outline_mesh_vbo", "_intersection_vbo",
                    "_engraving_bond_vbo",
                ):
                    value = getattr(self, attribute)
                    if value:
                        glDeleteBuffers(1, [value])
                for attribute in (
                    "_atom_vao", "_opaque_vao", "_bond_vao", "_cell_vao",
                    "_face_vao", "_face_edge_vao",
                    "_atom_outline_vao", "_intersection_vao",
                    "_engraving_bond_vao",
                ):
                    value = getattr(self, attribute)
                    if value:
                        glDeleteVertexArrays(1, [value])
                for attribute in ("_program", "_atom_program", "_edge_program", "_atom_outline_program", "_bond_program", "_triangle_program", "_segment_program"):
                    value = getattr(self, attribute)
                    if value:
                        glDeleteProgram(value)
        finally:
            if can_delete:
                self.doneCurrent()
            self._disconnect_context_cleanup()
            self._reset_gpu_objects()
            self._cleaning_up = False

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)


__all__ = ["OpenGLUnitCellViewer"]
