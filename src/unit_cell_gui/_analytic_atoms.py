"""Analytic atom sphere impostors; per-sample depth for orthogonal views.
Legacy mesh handles general transforms and contexts without sample shading.
"""
VERTEX_SHADER = r"""
#version 400 core
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

sample out vec2 v_offset;
flat out vec2 v_depth;
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
    v_offset = vec2((gl_VertexID & 1) == 0 ? -1.0 : 1.0,
                    (gl_VertexID & 2) == 0 ? -1.0 : 1.0);
    vec3 camera_position = u_orientation * a_center_radius.xyz;
    vec4 center = u_projection * vec4(camera_position, 1.0);
    v_depth = vec2(center.z, u_projection[2][2] * a_center_radius.w);
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
    // Clip the analytic surface, not the center plane of the quad.
    vec2 half_size = vec2(u_projection[0][0], u_projection[1][1]) * a_center_radius.w;
    gl_Position = a_center_radius.w <= 0.0 ? vec4(2.0, 2.0, 2.0, 1.0)
        : vec4(center.xy + v_offset * half_size, 0.0, 1.0);
}
"""
FRAGMENT_SHADER = r"""
#version 400 core
sample in vec2 v_offset;
flat in vec2 v_depth;
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
    float rho2 = dot(v_offset, v_offset);
    if (rho2 > 1.0) discard;
    float z = sqrt(max(0.0, 1.0 - rho2));
    float depth = v_depth.x + v_depth.y * z;
    if (depth > 1.0) {
        z = -z;
        depth = v_depth.x + v_depth.y * z;
    }
    if (depth < -1.0 || depth > 1.0) discard;
    gl_FragDepth = 0.5 * (depth + 1.0);
    vec3 normal = vec3(v_offset, z);

    // Camera-facing occupancy pie.  Because the analytic normal is in camera
    // space, the sector layout stays fixed on screen while the crystal turns.
    float tau = 6.283185307179586;
    float pie_position = 0.0;
    if (v_cumulative0.x < 1.0 && rho2 > 0.0)
        pie_position = fract(atan(normal.x, normal.y) / tau + 1.0);
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

def shader_sources(core_40):
    if core_40:
        return VERTEX_SHADER, FRAGMENT_SHADER
    header = "#version 330 core\n#extension GL_ARB_gpu_shader5 : require\n#extension GL_ARB_sample_shading : require"
    return tuple(source.replace("#version 400 core", header) for source in (VERTEX_SHADER, FRAGMENT_SHADER))
