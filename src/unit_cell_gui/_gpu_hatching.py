"""Compact view-dependent face plans. The GPU clips and expands each stroke.

No individual stroke endpoints are built by Python. The original implementation
remains the reference and fallback. All direction/family choices use float64.
"""
import numpy as np
from .hatching import propagated_hatch_directions, quadrilateral_hatch_line_count
from .view_helpers import (_is_octahedron, _octahedron_face_families,
    _octahedron_opposite_vertex, _opposite_edge, _line_direction, _qpoint_array)


def face_plans(polyhedron, vertices, projected, visible, normals, divisions):
    if not visible:
        return []
    anchor = max(visible, key=lambda i: float(normals[i][2]))
    face_mode = float(normals[anchor][2]) >= .999
    plans = {}
    def add(i, values, start, step, count, direction, tolerance, mode=0):
        # Centre scalar values locally, then scale to a range of order one.
        values = np.asarray(values, float)
        origin = float(values[0]); span = float(np.ptp(values))
        if span <= tolerance or count <= 0:
            return
        direction = np.asarray(direction, float)
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            return
        plans[i] = ((values-origin)/span, (start-origin)/span, step/span,
                    count, direction/length, tolerance/span, mode)
    if not _is_octahedron(polyhedron):
        directions = propagated_hatch_directions(polyhedron.faces, vertices, visible,
                                                  {anchor} if face_mode else ())
        for i, direction in directions.items():
            face = polyhedron.faces[i]; pts = vertices[list(face)]
            across = np.cross(normals[i], direction)
            length = float(np.linalg.norm(across))
            if length < 1e-12: continue
            values = (pts-pts[0]) @ (across/length)
            tol = max(1e-12, float(np.max(np.linalg.norm(pts-pts[0], axis=1)))*1e-10)
            count = quadrilateral_hatch_line_count(divisions) if len(face)==4 else divisions
            step = float(np.ptp(values))/(count+1.)
            add(i, values, float(values.min())+step, step, count, direction, tol)
    elif face_mode:
        _, families, references = _octahedron_face_families(polyhedron, visible, anchor)
        by_family = {}
        for i, family in families.items(): by_family.setdefault(family, []).append(i)
        for family, ids in by_family.items():
            a,b = references[family]
            direction = _line_direction(projected[a], projected[b])
            normal = np.array([-direction[1],direction[0]])
            values = np.asarray([_qpoint_array(p) for p in projected]) @ normal
            span = float(np.ptp(values[[v for i in ids for v in polyhedron.faces[i]]]))
            if span <= 1e-9: continue
            step = span/(divisions+1.)
            ref = .5*(values[a]+values[b]); start = ref + (-divisions-2)*step
            for i in ids:
                add(i, values[list(polyhedron.faces[i])], start, step, 2*divisions+5,
                    [direction[0],-direction[1],0.], 1e-8, 1)
    else:
        nearest = int(np.argmax(vertices[:,2]))
        opposite = _octahedron_opposite_vertex(polyhedron, nearest)
        for i in visible:
            face = polyhedron.faces[i]
            apex = nearest if nearest in face else opposite if opposite in face else max(face,key=lambda v:float(vertices[v,2]))
            a,b = _opposite_edge(face,int(apex),projected)
            # In this branch every face is triangular; interpolation is identical
            # to the legacy _single_face_hatches triangle branch.
            values = np.array([1. if v in (a,b) else 0. for v in face])
            step = 1./(divisions+1.)
            add(i, values, step, step, divisions, vertices[b]-vertices[a], 1e-12)
    return [(i,plans[i]) for i in visible if i in plans]


def pack_plans(polyhedron, vertices, projected, visible, normals, divisions, lift):
    plans = face_plans(polyhedron, vertices, projected, visible, normals, divisions)
    # Three RGBA texels per face; then one per vertex. Geometry is in camera
    # coordinates, already lifted by the original normal-dependent offset.
    headers = []; payload = []; start = 3*len(plans); maximum = 0
    for i,(values,first,step,count,direction,tolerance,mode) in plans:
        pts = vertices[list(polyhedron.faces[i])] + normals[i]*lift
        # Precision-sensitive configurations use the CPU reference instead.
        if not np.isfinite(pts).all() or np.max(np.abs(pts)) > 1e30:
            raise ValueError('hatch coordinates outside GPU range')
        headers.extend([[start,len(pts),count,first], [step,tolerance,mode,0], [*direction,0]])
        payload.extend(np.column_stack((pts,values)).tolist())
        start += len(pts); maximum=max(maximum,count)
    return np.asarray(headers+payload,dtype=np.float32).reshape(-1,4), len(plans), maximum


VERTEX_SHADER = '''#version 330 core
uniform samplerBuffer u_hatch_records;
uniform mat4 u_projection;
out vec4 v_colour;
void main() {
    v_colour=vec4(vec3(30.0/255.0),1.0);
    gl_Position=vec4(2.0,2.0,2.0,1.0);
    int h=3*gl_InstanceID;
    vec4 meta=texelFetch(u_hatch_records,h);
    vec4 params=texelFetch(u_hatch_records,h+1);
    vec3 dir=texelFetch(u_hatch_records,h+2).xyz;
    int stroke=gl_VertexID/2;
    if (stroke>=int(meta.z)) return;
    int start=int(meta.x), n=int(meta.y);
    float level=meta.w+float(stroke)*params.x;
    float eps=max(params.y, 2.0e-7);
    vec4 p=texelFetch(u_hatch_records,start+n-1);
    vec3 origin=texelFetch(u_hatch_records,start).xyz;
    vec3 first=vec3(0),last=vec3(0);
    float lo=0.0,hi=0.0;
    bool found=false;
    for(int k=0;k<n;++k){
        vec4 q=texelFetch(u_hatch_records,start+k);
        float a=p.w-level,b=q.w-level;
        // At most two candidate points per edge: vertex and crossing.
        for(int j=0;j<2;++j){
            vec3 hit;
            if(j==0){ if(abs(a)>eps) continue; hit=p.xyz; }
            else {
                // The small normalized tolerance avoids duplicating a vertex
                // which is already accepted as on the line.
                if(!((a < -eps && b > eps)||(a > eps && b < -eps))) continue;
                hit=mix(p.xyz,q.xyz,a/(a-b));
            }
            float t=dot(hit-origin,dir);
            if(!found || t<lo){lo=t;first=hit;}
            if(!found || t>hi){hi=t;last=hit;}
            found=true;
        }
        p=q;
    }
    if(!found || hi<=lo) return;
    gl_Position=u_projection*vec4((gl_VertexID%2==0)?first:last,1.0);
}
'''
