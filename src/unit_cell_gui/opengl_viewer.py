"""Cached OpenGL 3.3 unit-cell renderer.

Geometry is generated only when the scene/style changes and uploaded to static
GPU buffers.  Interactive rotation changes only one orientation uniform; no
sphere/cylinder meshes are regenerated and no vertex buffers are re-uploaded
while the user drags the structure.
"""

from __future__ import annotations

import ctypes
import math
import time
from collections import deque

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
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
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
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
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetProgramInfoLog,
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
    glViewport,
)
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPainter, QPen, QPolygonF, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy

from .gl_geometry import cylinder_mesh, line_vertices, polygon_triangles, sphere_mesh
from .hatching import hatch_division_count
from .models import DisplayOptions, Scene, StyleOverrides
from .view_helpers import _polyhedron_hatches, screen_drag_orientation


# One static vertex format is shared by every buffer:
# position.xyz, normal.xyz, colour.rgba, lit
_VERTEX_FLOATS = 11
_VERTEX_STRIDE = _VERTEX_FLOATS * 4

_ATOM_MAX_COMPONENTS = 8
_ATOM_VERTEX_FLOATS = 3 + 3 + 3 * _ATOM_MAX_COMPONENTS + _ATOM_MAX_COMPONENTS
_ATOM_VERTEX_STRIDE = _ATOM_VERTEX_FLOATS * 4


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
    vec3 camera_position = u_orientation * a_position;
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
    gl_Position = u_projection * vec4(camera_position, 1.0);
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

_EDGE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 2) in vec4 a_colour;

uniform mat4 u_projection;
uniform mat3 u_orientation;

out vec4 v_colour;

void main() {
    vec3 camera_position = u_orientation * a_position;
    v_colour = a_colour;
    gl_Position = u_projection * vec4(camera_position, 1.0);
}
"""


_EDGE_GEOMETRY_SHADER = """
#version 330 core
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in vec4 v_colour[];
out vec4 g_colour;

uniform vec2 u_viewport;
uniform float u_edge_width;
uniform float u_depth_bias;

void emit_edge_vertex(vec4 clip, vec2 ndc_offset, vec4 colour) {
    vec4 shifted = clip;
    shifted.xy += ndc_offset * clip.w;
    shifted.z += u_depth_bias * clip.w;
    gl_Position = shifted;
    g_colour = colour;
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

    // Expand by half the requested width at both ends too, so neighbouring
    // edge quads overlap slightly instead of leaving tiny corner gaps.
    vec2 end_extend_ndc = vec2(
        direction_pixels.x * half_width * 2.0 / u_viewport.x,
        direction_pixels.y * half_width * 2.0 / u_viewport.y
    );
    vec2 side_ndc = vec2(
        perpendicular_pixels.x * half_width * 2.0 / u_viewport.x,
        perpendicular_pixels.y * half_width * 2.0 / u_viewport.y
    );

    vec2 start = -end_extend_ndc;
    vec2 finish = end_extend_ndc;
    emit_edge_vertex(p0, start + side_ndc, v_colour[0]);
    emit_edge_vertex(p0, start - side_ndc, v_colour[0]);
    emit_edge_vertex(p1, finish + side_ndc, v_colour[1]);
    emit_edge_vertex(p1, finish - side_ndc, v_colour[1]);
    EndPrimitive();
}
"""


_EDGE_FRAGMENT_SHADER = """
#version 330 core
in vec4 g_colour;
out vec4 frag_colour;

void main() {
    frag_colour = g_colour;
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


def _atom_vertex_block(
    positions: np.ndarray,
    normals: np.ndarray,
    colours: np.ndarray,
    cumulative: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
    colours = np.asarray(colours, dtype=np.float32).reshape(_ATOM_MAX_COMPONENTS, 3)
    cumulative = np.asarray(cumulative, dtype=np.float32).reshape(_ATOM_MAX_COMPONENTS)
    if positions.shape != normals.shape:
        raise ValueError("positions and normals must have matching shapes")
    count = positions.shape[0]
    if count == 0:
        return np.empty((0, _ATOM_VERTEX_FLOATS), dtype=np.float32)
    colour_values = np.broadcast_to(colours.reshape(1, -1), (count, 3 * _ATOM_MAX_COMPONENTS))
    cumulative_values = np.broadcast_to(cumulative.reshape(1, -1), (count, _ATOM_MAX_COMPONENTS))
    return np.ascontiguousarray(
        np.concatenate((positions, normals, colour_values, cumulative_values), axis=1),
        dtype=np.float32,
    )


def _combine_atom_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    nonempty = [block for block in blocks if block.size]
    if not nonempty:
        return np.empty((0, _ATOM_VERTEX_FLOATS), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(nonempty, axis=0), dtype=np.float32)


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
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSamples(4)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.setFormat(fmt)

        self.scene: Scene | None = None
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
        self.opengl_error: str | None = None

        self._atom_vao = 0
        self._atom_vbo = 0
        self._atom_count = 0
        self._opaque_vao = 0
        self._opaque_vbo = 0
        self._opaque_count = 0
        self._bond_vao = 0
        self._bond_vbo = 0
        self._bond_count = 0
        self._bond_dirty = True
        self._cell_vao = 0
        self._cell_vbo = 0
        self._cell_count = 0
        self._face_vao = 0
        self._face_vbo = 0
        self._face_count = 0
        self._face_ranges: list[tuple[np.ndarray, int, int]] = []
        self._face_edge_vao = 0
        self._face_edge_vbo = 0
        self._face_edge_count = 0
        self._face_edge_ranges: list[tuple[np.ndarray, int, int]] = []
        self._hatch_vao = 0
        self._hatch_vbo = 0
        self._hatch_count = 0
        self._hatch_dirty = True
        self._atom_outline_vao = 0
        self._atom_outline_vbo = 0
        self._atom_outline_count = 0
        self._atom_outline_dirty = True
        self._intersection_vao = 0
        self._intersection_vbo = 0
        self._intersection_count = 0
        self._engraving_bond_vao = 0
        self._engraving_bond_vbo = 0
        self._engraving_bond_count = 0
        self._gpu_dirty = True
        self._atom_dirty = True
        self._bond_dirty = True
        self._intersection_dirty = True

        # Expensive tessellation depends on the Scene, not on most display
        # switches.  Keep it on the CPU and reuse it when the host toggles
        # polyhedra, bonds, engraving, etc.
        self._atom_mesh_cache: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
        self._bond_mesh_cache: dict[tuple[int, float], tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {}
        self._poly_face_mesh_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        self._show_fps = False
        self._frame_times: deque[float] = deque(maxlen=90)

        self.setMinimumSize(420, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

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

    def _clear_mesh_caches(self, *, atoms_only: bool = False) -> None:
        self._atom_mesh_cache.clear()
        if not atoms_only:
            self._bond_mesh_cache.clear()
            self._poly_face_mesh_cache.clear()

    def set_scene(self, scene: Scene | None, *, reset_camera: bool = True) -> None:
        if scene is not None and not isinstance(scene, Scene):
            raise TypeError("scene must be a unit_cell_gui.Scene or None")
        self.scene = scene
        self._clear_mesh_caches()
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
        self._atom_outline_dirty = True
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
        if changed & {"engraving", "show_external_atoms"}:
            self._atom_dirty = True
            self._atom_outline_dirty = True
        if "show_external_atoms" in changed:
            self._bond_dirty = True
        if "atom_scale" in changed:
            self._clear_mesh_caches(atoms_only=True)
            self._atom_dirty = True
            self._atom_outline_dirty = True
            self._intersection_dirty = True
        # Intersection geometry itself is independent of colour/BW mode and of
        # the global show/hide switches.  Build it lazily only when both layers
        # are actually visible.
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
        poly_style_changed = (
            new_poly_colours != self.polyhedron_site_colours
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
        if atom_visibility_changed or poly_visibility_changed:
            self._intersection_dirty = True
        if poly_visibility_changed or poly_style_changed:
            self._gpu_dirty = True
            self._hatch_dirty = True
        self.update()

    def zoom_by(self, steps: float) -> None:
        self.zoom = float(np.clip(self.zoom * (1.12 ** float(steps)), 0.25, 4.5))
        self._hatch_dirty = True
        self._atom_outline_dirty = True
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
        radius = self.scene.base_radius if self.scene is not None else 1.0
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
        radius = self.scene.base_radius if self.scene is not None else 1.0
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
        radius = self.scene.base_radius if self.scene is not None else 1.0
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

    @staticmethod
    def _create_atom_buffer() -> tuple[int, int]:
        vao = int(glGenVertexArrays(1))
        vbo = int(glGenBuffers(1))
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, _ATOM_VERTEX_STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, _ATOM_VERTEX_STRIDE, ctypes.c_void_p(3 * 4))
        offset = 6
        for location in range(2, 10):
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(
                location, 3, GL_FLOAT, GL_FALSE, _ATOM_VERTEX_STRIDE, ctypes.c_void_p(offset * 4)
            )
            offset += 3
        glEnableVertexAttribArray(10)
        glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, _ATOM_VERTEX_STRIDE, ctypes.c_void_p(offset * 4))
        offset += 4
        glEnableVertexAttribArray(11)
        glVertexAttribPointer(11, 4, GL_FLOAT, GL_FALSE, _ATOM_VERTEX_STRIDE, ctypes.c_void_p(offset * 4))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return vao, vbo

    @staticmethod
    def _upload_static(vao: int, vbo: int, vertices: np.ndarray) -> int:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        return int(vertices.shape[0])

    def initializeGL(self) -> None:
        try:
            self._program = _create_program()
            self._atom_program = _create_program(_ATOM_VERTEX_SHADER, _ATOM_FRAGMENT_SHADER)
            self._edge_program = _create_program(
                _EDGE_VERTEX_SHADER,
                _EDGE_FRAGMENT_SHADER,
                _EDGE_GEOMETRY_SHADER,
            )
            self._atom_vao, self._atom_vbo = self._create_atom_buffer()
            self._opaque_vao, self._opaque_vbo = self._create_buffer()
            self._bond_vao, self._bond_vbo = self._create_buffer()
            self._cell_vao, self._cell_vbo = self._create_buffer()
            self._face_vao, self._face_vbo = self._create_buffer()
            self._face_edge_vao, self._face_edge_vbo = self._create_buffer()
            self._hatch_vao, self._hatch_vbo = self._create_buffer()
            self._atom_outline_vao, self._atom_outline_vbo = self._create_buffer()
            self._intersection_vao, self._intersection_vbo = self._create_buffer()
            self._engraving_bond_vao, self._engraving_bond_vbo = self._create_buffer()
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
        self._atom_outline_dirty = True

    def _restore_render_state(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_GREATER)
        glClearDepth(0.0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)

    def _cached_atom_mesh(self, index: int, atom, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = (int(index), round(float(self.atom_scale), 10))
        mesh = self._atom_mesh_cache.get(key)
        if mesh is None:
            mesh = sphere_mesh(center, self._atom_radius(atom))
            self._atom_mesh_cache[key] = mesh
        return mesh

    def _cached_bond_mesh(self, index: int, first: np.ndarray, second: np.ndarray, radius: float):
        key = (int(index), round(float(radius), 10))
        mesh = self._bond_mesh_cache.get(key)
        if mesh is None:
            midpoint = 0.5 * (first + second)
            mesh = (
                cylinder_mesh(first, midpoint, radius),
                cylinder_mesh(midpoint, second, radius),
            )
            self._bond_mesh_cache[key] = mesh
        return mesh

    def _cached_poly_face_mesh(self, poly_index: int, face_index: int, polyhedron, face):
        key = (int(poly_index), int(face_index))
        cached = self._poly_face_mesh_cache.get(key)
        if cached is None:
            world = np.asarray(polyhedron.vertices[list(face)], dtype=float)
            positions, normals = polygon_triangles(world, polyhedron.center)
            cached = (world, positions, normals)
            self._poly_face_mesh_cache[key] = cached
        return cached

    def _build_atom_geometry(self) -> np.ndarray:
        if self.scene is None:
            return np.empty((0, _ATOM_VERTEX_FLOATS), dtype=np.float32)
        blocks: list[np.ndarray] = []
        for atom_index, (atom, center) in enumerate(zip(self.scene.atoms, self.scene.atom_centers)):
            if not self._atom_visible(atom):
                continue
            if atom.external and not self.show_external_atoms:
                continue
            positions, normals = self._cached_atom_mesh(atom_index, atom, center)
            colours, cumulative = self._atom_pie_material(atom)
            blocks.append(_atom_vertex_block(positions, normals, colours, cumulative))
        return _combine_atom_blocks(blocks)

    def _build_colour_bond_geometry(self) -> np.ndarray:
        if self.scene is None:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
        blocks: list[np.ndarray] = []
        bond_radius = max(0.018, 0.020 * float(self.scene.base_radius)) / 3.0
        for bond_index, (first, second) in enumerate(self.scene.bonds):
            atom_a = self.scene.atoms[int(first)]
            atom_b = self.scene.atoms[int(second)]
            if not self._atom_visible(atom_a) or not self._atom_visible(atom_b):
                continue
            if (atom_a.external or atom_b.external) and not self.show_external_atoms:
                continue
            a = np.asarray(self.scene.atom_centers[int(first)], dtype=float)
            b = np.asarray(self.scene.atom_centers[int(second)], dtype=float)
            halves = self._cached_bond_mesh(bond_index, a, b, bond_radius)
            for (positions, normals), colour in zip(
                halves,
                (self._dominant_atom_colour_coloured(atom_a), self._dominant_atom_colour_coloured(atom_b)),
            ):
                blocks.append(_vertex_block(positions, normals, colour, lit=True))
        return _combine_blocks(blocks)

    def _build_static_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, int, int]], np.ndarray, list[tuple[np.ndarray, int, int]]]:
        if self.scene is None:
            empty = np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
            atom_empty = np.empty((0, _ATOM_VERTEX_FLOATS), dtype=np.float32)
            return atom_empty, empty, empty.copy(), empty.copy(), [], empty.copy(), []

        atom_blocks: list[np.ndarray] = []
        opaque_blocks: list[np.ndarray] = []
        cell_blocks: list[np.ndarray] = []
        face_blocks: list[np.ndarray] = []
        face_ranges: list[tuple[np.ndarray, int, int]] = []
        face_edge_blocks: list[np.ndarray] = []
        face_edge_ranges: list[tuple[np.ndarray, int, int]] = []

        if self.show_cell:
            colour = QColor(90, 90, 90) if not self.engraving else QColor(60, 60, 60)
            for first, second in self.scene.cell_edges:
                positions, normals = line_vertices(
                    self.scene.cell_vertices[int(first)],
                    self.scene.cell_vertices[int(second)],
                )
                cell_blocks.append(_vertex_block(positions, normals, colour, lit=False))

        if self.scene.polyhedra:
            transparent_offset = 0
            transparent_edge_offset = 0
            edge_lift = max(1.0e-4, 8.0e-4 * float(self.scene.base_radius))
            for poly_index, polyhedron in enumerate(self.scene.polyhedra):
                if not self.polyhedron_site_visibility.get(polyhedron.site_key, True):
                    continue
                colour = QColor(
                    self.polyhedron_site_colours.get(
                        polyhedron.site_key,
                        _blend_components(polyhedron.components).name(),
                    )
                )
                if self.engraving:
                    colour = QColor(245, 245, 245)
                occupancy = min(
                    1.0,
                    sum(max(0.0, float(component.occupancy)) for component in polyhedron.components),
                )
                opaque = self.engraving or (polyhedron.site_key in self.polyhedron_opaque_sites)
                alpha = 1.0 if opaque else (0.28 + 0.36 * occupancy)
                edge_colour = QColor(20, 20, 20) if self.engraving else _polyhedron_edge_colour(colour)
                poly_center = np.asarray(polyhedron.center, dtype=float)
                edge_faces: dict[tuple[int, int], list[np.ndarray]] = {}
                for face_index, face in enumerate(polyhedron.faces):
                    world, positions, normals = self._cached_poly_face_mesh(
                        poly_index, face_index, polyhedron, face
                    )
                    block = _vertex_block(positions, normals, colour, alpha=alpha, lit=False)
                    if opaque:
                        opaque_blocks.append(block)
                    else:
                        face_blocks.append(block)
                        face_ranges.append((world.mean(axis=0), transparent_offset, block.shape[0]))
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
                    edge_positions, edge_vertex_normals = line_vertices(
                        first_point + outward, second_point + outward
                    )
                    edge_block = _vertex_block(
                        edge_positions,
                        edge_vertex_normals,
                        edge_colour,
                        alpha=1.0 if self.engraving else alpha,
                        lit=False,
                    )
                    face_edge_blocks.append(edge_block)
                    midpoint = 0.5 * (first_point + second_point)
                    face_edge_ranges.append((midpoint, transparent_edge_offset, edge_block.shape[0]))
                    transparent_edge_offset += edge_block.shape[0]

        return (
            _combine_atom_blocks(atom_blocks),
            _combine_blocks(opaque_blocks),
            _combine_blocks(cell_blocks),
            _combine_blocks(face_blocks),
            face_ranges,
            _combine_blocks(face_edge_blocks),
            face_edge_ranges,
        )

    def _build_hatch_geometry(self) -> np.ndarray:
        if self.scene is None or not self.engraving or not self.hatching:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)

        selected: list[tuple[object, np.ndarray]] = []
        for polyhedron in self.scene.polyhedra:
            if not self.polyhedron_site_visibility.get(polyhedron.site_key, True):
                continue
            center_camera = self.orientation @ np.asarray(polyhedron.center, dtype=float)
            selected.append((polyhedron, center_camera))
        if not selected:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)

        depths = [float(center[2]) for _polyhedron, center in selected]
        minimum, maximum = min(depths), max(depths)
        hatch_blocks: list[np.ndarray] = []
        inverse_orientation = np.linalg.inv(np.asarray(self.orientation, dtype=float))
        hatch_colour = QColor(30, 30, 30)
        hatch_lift = max(1.0e-4, 1.0e-3 * float(self.scene.base_radius))

        for polyhedron, center_camera in selected:
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
                continue
            hatch_far = (
                0.5
                if abs(maximum - minimum) < 1e-12
                else (maximum - float(center_camera[2])) / (maximum - minimum)
            )
            hatches = _polyhedron_hatches(
                polyhedron,
                vertices_camera,
                projected,
                visible_faces,
                normals,
                hatch_division_count(hatch_far, self.hatch_density, self.hatch_grip),
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

    def _build_engraving_bond_geometry(self) -> np.ndarray:
        if self.scene is None:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
        blocks: list[np.ndarray] = []
        colour = QColor("#777777")
        for first, second in self.scene.bonds:
            atom_a = self.scene.atoms[int(first)]
            atom_b = self.scene.atoms[int(second)]
            if not self._atom_visible(atom_a) or not self._atom_visible(atom_b):
                continue
            if (atom_a.external or atom_b.external) and not self.show_external_atoms:
                continue
            positions, normals = line_vertices(
                self.scene.atom_centers[int(first)],
                self.scene.atom_centers[int(second)],
            )
            blocks.append(_vertex_block(positions, normals, colour, lit=False))
        return _combine_blocks(blocks)

    def _build_atom_outline_geometry(self) -> np.ndarray:
        if self.scene is None or not self.show_atoms:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
        orientation = np.asarray(self.orientation, dtype=float)
        inverse_orientation = np.linalg.inv(orientation)
        blocks: list[np.ndarray] = []
        segments = 64
        for atom, center in zip(self.scene.atoms, self.scene.atom_centers):
            if not self._atom_visible(atom):
                continue
            if atom.external and not self.show_external_atoms:
                continue
            occupancy = sum(max(0.0, float(component.occupancy)) for component in atom.components)
            partial = self.engraving and occupancy < 1.0 - 1.0e-8
            colour = QColor(30, 30, 30) if self.engraving else _blend_components(atom.components).darker(165)
            centre_camera = orientation @ np.asarray(center, dtype=float)
            # Put the ring just outside the sphere silhouette by half the old
            # 0.9 px cosmetic pen width.  This avoids the sphere depth buffer
            # eating the inner half of its own outline.
            radius = self._atom_radius(atom) + 0.45 / max(self._screen_scale(), 1e-12)
            points: list[np.ndarray] = []
            normals: list[np.ndarray] = []
            for index in range(segments):
                if partial and index % 2:
                    continue
                start_angle = 2.0 * math.pi * index / segments
                end_angle = 2.0 * math.pi * (index + (0.58 if partial else 1.0)) / segments
                first_camera = centre_camera + radius * np.array(
                    [math.cos(start_angle), math.sin(start_angle), 0.0], dtype=float
                )
                second_camera = centre_camera + radius * np.array(
                    [math.cos(end_angle), math.sin(end_angle), 0.0], dtype=float
                )
                first_world = inverse_orientation @ first_camera
                second_world = inverse_orientation @ second_camera
                segment_positions, segment_normals = line_vertices(first_world, second_world)
                points.append(segment_positions)
                normals.append(segment_normals)
            if points:
                blocks.append(
                    _vertex_block(
                        np.concatenate(points, axis=0),
                        np.concatenate(normals, axis=0),
                        colour,
                        lit=False,
                    )
                )
        return _combine_blocks(blocks)

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

    def _build_atom_face_intersection_geometry(self) -> np.ndarray:
        """Build the actual sphere/face intersection curves.

        Each polyhedron face is a finite planar polygon.  A sphere/plane
        intersection is a circle; only the circle pieces that lie inside the
        finite face polygon are emitted.  Hidden parts are then rejected by the
        normal OpenGL depth test.
        """
        if self.scene is None:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)

        blocks: list[np.ndarray] = []
        samples = 160
        intersection_colour = QColor(20, 20, 20)

        visible_atoms = [
            (
                atom,
                np.asarray(center, dtype=float),
                self._atom_radius(atom),
                intersection_colour,
            )
            for atom, center in zip(self.scene.atoms, self.scene.atom_centers)
            if self._atom_visible(atom)
            and (self.show_external_atoms or not atom.external)
        ]
        if not visible_atoms:
            return np.empty((0, _VERTEX_FLOATS), dtype=np.float32)
        atom_centres = np.asarray([item[1] for item in visible_atoms], dtype=float)
        atom_radii = np.asarray([item[2] for item in visible_atoms], dtype=float)

        for polyhedron in self.scene.polyhedra:
            if not self.polyhedron_site_visibility.get(polyhedron.site_key, True):
                continue
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
                    _atom, atom_center, atom_radius, colour = visible_atoms[int(atom_index)]
                    signed_distance = float(signed_all[int(atom_index)])
                    circle_radius_sq = atom_radius * atom_radius - signed_distance * signed_distance
                    if circle_radius_sq <= 1.0e-12:
                        continue
                    circle_radius = math.sqrt(circle_radius_sq)
                    circle_center = atom_center - signed_distance * normal

                    points = [
                        circle_center
                        + circle_radius
                        * (
                            math.cos(2.0 * math.pi * index / samples) * u
                            + math.sin(2.0 * math.pi * index / samples) * v
                        )
                        for index in range(samples)
                    ]

                    positions: list[np.ndarray] = []
                    normals: list[np.ndarray] = []
                    for index, first in enumerate(points):
                        second = points[(index + 1) % samples]
                        midpoint = 0.5 * (first + second)
                        if not self._point_in_convex_face(midpoint, polygon, normal):
                            continue
                        segment_positions, segment_normals = line_vertices(first, second)
                        positions.append(segment_positions)
                        normals.append(segment_normals)
                    if positions:
                        blocks.append(
                            _vertex_block(
                                np.concatenate(positions, axis=0),
                                np.concatenate(normals, axis=0),
                                colour,
                                lit=False,
                            )
                        )

        return _combine_blocks(blocks)

    def _rebuild_gpu_cache(self) -> None:
        _atoms, opaque, cell, faces, ranges, face_edges, face_edge_ranges = self._build_static_geometry()
        self._opaque_count = self._upload_static(self._opaque_vao, self._opaque_vbo, opaque)
        self._cell_count = self._upload_static(self._cell_vao, self._cell_vbo, cell)
        self._face_count = self._upload_static(self._face_vao, self._face_vbo, faces)
        self._face_ranges = ranges
        self._face_edge_count = self._upload_static(self._face_edge_vao, self._face_edge_vbo, face_edges)
        self._face_edge_ranges = face_edge_ranges
        self._gpu_dirty = False

    def _rebuild_bond_cache(self) -> None:
        colour_bonds = self._build_colour_bond_geometry()
        self._bond_count = self._upload_static(self._bond_vao, self._bond_vbo, colour_bonds)
        engraving_bonds = self._build_engraving_bond_geometry()
        self._engraving_bond_count = self._upload_static(
            self._engraving_bond_vao, self._engraving_bond_vbo, engraving_bonds
        )
        self._bond_dirty = False

    def _rebuild_atom_cache(self) -> None:
        atoms = self._build_atom_geometry()
        self._atom_count = self._upload_static(self._atom_vao, self._atom_vbo, atoms)
        self._atom_dirty = False

    def _rebuild_intersection_cache(self) -> None:
        intersections = self._build_atom_face_intersection_geometry()
        self._intersection_count = self._upload_static(
            self._intersection_vao, self._intersection_vbo, intersections
        )
        self._intersection_dirty = False

    def _rebuild_hatch_cache(self) -> None:
        if not self.context() or not self.context().isValid():
            return
        hatches = self._build_hatch_geometry()
        self._hatch_count = self._upload_static(self._hatch_vao, self._hatch_vbo, hatches)
        self._hatch_dirty = False

    def _rebuild_atom_outline_cache(self) -> None:
        if not self.context() or not self.context().isValid():
            return
        outlines = self._build_atom_outline_geometry()
        self._atom_outline_count = self._upload_static(
            self._atom_outline_vao, self._atom_outline_vbo, outlines
        )
        self._atom_outline_dirty = False

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
        device_ratio = max(1.0, float(self.devicePixelRatioF()))
        glUniform2f(
            self._edge_uniform_viewport,
            max(1.0, float(self.width()) * device_ratio),
            max(1.0, float(self.height()) * device_ratio),
        )
        glUniform1f(self._edge_uniform_width, float(width_px) * device_ratio)
        glUniform1f(self._edge_uniform_depth_bias, float(depth_bias))

    @staticmethod
    def _draw_buffer(vao: int, count: int, mode: int) -> None:
        if count <= 0:
            return
        glBindVertexArray(vao)
        glDrawArrays(mode, 0, count)
        glBindVertexArray(0)

    def _ordered_face_ranges(self) -> list[tuple[np.ndarray, int, int]]:
        if not self._face_ranges:
            return []
        orientation = np.asarray(self.orientation, dtype=float)
        return sorted(
            self._face_ranges,
            key=lambda item: float((orientation @ item[0])[2]),
        )

    def _draw_transparent_faces(self) -> None:
        if not self.show_polyhedra or self._face_count <= 0 or not self._face_ranges:
            return
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._face_vao)
        # Only tiny face-centre transforms and draw calls remain during rotation;
        # vertex data itself stays resident on the GPU.
        for _centre, first, count in self._ordered_face_ranges():
            glDrawArrays(GL_TRIANGLES, int(first), int(count))
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _draw_face_edges(self) -> None:
        if not self.show_polyhedra or self._face_edge_count <= 0 or not self._face_edge_ranges:
            return
        self._set_edge_uniforms(width_px=1.575, depth_bias=1.0e-5)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._face_edge_vao)
        orientation = np.asarray(self.orientation, dtype=float)
        ordered = sorted(
            self._face_edge_ranges,
            key=lambda item: float((orientation @ item[0])[2]),
        )
        for _centre, first, count in ordered:
            glDrawArrays(GL_LINES, int(first), int(count))
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _draw_hatches(self) -> None:
        if not self.show_polyhedra or not self.engraving or not self.hatching or self._hatch_count <= 0:
            return
        self._set_edge_uniforms(width_px=0.75, depth_bias=1.5e-5)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._hatch_vao)
        glDrawArrays(GL_LINES, 0, int(self._hatch_count))
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def _draw_engraving_bonds(self) -> None:
        if not self.show_bonds or not self.engraving or self._engraving_bond_count <= 0:
            return
        self._set_edge_uniforms(width_px=1.8, depth_bias=0.0)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._engraving_bond_vao)
        glDrawArrays(GL_LINES, 0, int(self._engraving_bond_count))
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)

    def _draw_atom_outlines(self) -> None:
        if not self.show_atoms or self._atom_outline_count <= 0:
            return
        width_px = 0.9 if self.engraving else 0.95
        bias = 2.0e-5 if self.engraving else 1.4e-5
        self._set_edge_uniforms(width_px=width_px, depth_bias=bias)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._atom_outline_vao)
        glDrawArrays(GL_LINES, 0, int(self._atom_outline_count))
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)

    def _draw_atom_face_intersections(self) -> None:
        if not self.show_atoms or not self.show_polyhedra or self._intersection_count <= 0:
            return
        # The curve is geometrically coincident with both the sphere and the
        # face, so use only a tiny clip-space bias to make the ink stable.
        width_px = 1.0 if self.engraving else 0.95
        bias = 7.5e-6 if self.engraving else 8.5e-6
        self._set_edge_uniforms(width_px=width_px, depth_bias=bias)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._intersection_vao)
        glDrawArrays(GL_LINES, 0, int(self._intersection_count))
        glBindVertexArray(0)
        glDepthMask(GL_TRUE)

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
        if self.scene is None or not self.show_cell:
            return
        pen = QPen(QColor(75, 75, 75))
        pen.setWidthF(0.9)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for first, second in self.scene.cell_edges:
            painter.drawLine(
                self._project_world_point(self.scene.cell_vertices[int(first)]),
                self._project_world_point(self.scene.cell_vertices[int(second)]),
            )

    def _draw_overlay_basis(self, painter: QPainter) -> None:
        if self.scene is None or not self.show_basis:
            return
        directions = self.orientation @ (
            self.scene.basis_vectors / np.linalg.norm(self.scene.basis_vectors, axis=0)
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
        if self.scene is None or not self.show_atoms:
            return
        x, y = 12.0, 18.0
        metrics = QFontMetrics(painter.font())
        visible_elements = {
            component.element
            for atom in self.scene.atoms
            for component in atom.components
            if self._component_visible(component)
        }
        for element in self.scene.elements:
            if element not in visible_elements:
                continue
            component = next(
                component
                for atom in self.scene.atoms
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
            if self.scene is None:
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
        if self.opengl_error is None and self.scene is not None and self._program:
            if self._gpu_dirty:
                self._rebuild_gpu_cache()
            if self._atom_dirty:
                self._rebuild_atom_cache()
            if self._bond_dirty:
                self._rebuild_bond_cache()
            if self._intersection_dirty:
                self._rebuild_intersection_cache()
            if self._hatch_dirty:
                self._rebuild_hatch_cache()
            if self._atom_outline_dirty:
                self._rebuild_atom_outline_cache()
            self._set_frame_uniforms()
            if self.show_polyhedra:
                self._draw_buffer(self._opaque_vao, self._opaque_count, GL_TRIANGLES)
            if self.show_bonds and not self.engraving:
                self._draw_buffer(self._bond_vao, self._bond_count, GL_TRIANGLES)
            if self.show_atoms and self._atom_count > 0:
                self._set_atom_uniforms()
                self._draw_buffer(self._atom_vao, self._atom_count, GL_TRIANGLES)
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
            self._draw_transparent_faces()
            glUseProgram(0)
            self._draw_engraving_bonds()
            self._draw_hatches()
            self._draw_face_edges()
            glUseProgram(0)
        self._draw_overlay()

    def mousePressEvent(self, event) -> None:
        if (
            event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
            and self.scene is not None
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
                self._atom_outline_dirty = True
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

    def cleanup(self) -> None:
        if not self.context() or not self.context().isValid():
            return
        self.makeCurrent()
        try:
            for attribute in ("_atom_vbo", "_opaque_vbo", "_bond_vbo", "_cell_vbo", "_face_vbo", "_face_edge_vbo", "_hatch_vbo", "_atom_outline_vbo", "_intersection_vbo", "_engraving_bond_vbo"):
                value = getattr(self, attribute)
                if value:
                    glDeleteBuffers(1, [value])
                    setattr(self, attribute, 0)
            for attribute in ("_atom_vao", "_opaque_vao", "_bond_vao", "_cell_vao", "_face_vao", "_face_edge_vao", "_hatch_vao", "_atom_outline_vao", "_intersection_vao", "_engraving_bond_vao"):
                value = getattr(self, attribute)
                if value:
                    glDeleteVertexArrays(1, [value])
                    setattr(self, attribute, 0)
            if self._program:
                glDeleteProgram(self._program)
                self._program = 0
            if self._atom_program:
                glDeleteProgram(self._atom_program)
                self._atom_program = 0
            if self._edge_program:
                glDeleteProgram(self._edge_program)
                self._edge_program = 0
        finally:
            self.doneCurrent()

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)


__all__ = ["OpenGLUnitCellViewer"]
