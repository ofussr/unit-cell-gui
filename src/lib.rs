mod resources;
mod render;
mod frame;
mod contacts;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

type P3 = [f64; 3];
type P2 = [f64; 2];
type Bounds = [f64; 4];
struct Primitive {
    points: Vec<P3>,
    planes: Vec<P3>,
    depth: Option<P3>,
    edge: bool,
    bounds: Bounds,
}
fn read_f64(data: &[u8]) -> PyResult<Vec<f64>> {
    if data.len() % 8 != 0 {
        return Err(PyValueError::new_err("invalid float buffer"));
    }
    Ok(data
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect())
}
fn read_usize(data: &[u8]) -> PyResult<Vec<usize>> {
    if data.len() % 8 != 0 {
        return Err(PyValueError::new_err("invalid index buffer"));
    }
    data.chunks_exact(8)
        .map(|x| {
            usize::try_from(u64::from_le_bytes(x.try_into().unwrap()))
                .map_err(|_| PyValueError::new_err("index overflow"))
        })
        .collect()
}
fn overlaps(a: Bounds, b: Bounds, e: f64) -> bool {
    a[0] <= b[2] + e && b[0] <= a[2] + e && a[1] <= b[3] + e && b[1] <= a[3] + e
}
fn plane_value(p: P3, x: P2) -> f64 {
    p[0] * x[0] + p[1] * x[1] + p[2]
}
fn primitive(points: Vec<P3>, normal: P3, edge: bool, margin: f64, eps: f64) -> Primitive {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in &points {
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    let mut planes = Vec::new();
    let mut depth = None;
    if edge {
        b[0] -= margin;
        b[1] -= margin;
        b[2] += margin;
        b[3] += margin;
    } else {
        let pairs = points.iter().zip(points.iter().cycle().skip(1));
        let area: f64 = pairs.clone().map(|(p, q)| p[0] * q[1] - p[1] * q[0]).sum();
        let s = if area >= 0. { 1. } else { -1. };
        planes.extend(pairs.map(|(p, q)| {
            [
                s * (p[1] - q[1]),
                s * (q[0] - p[0]),
                s * (p[0] * q[1] - p[1] * q[0]),
            ]
        }));
        if normal[2].abs() > eps {
            let a = -normal[0] / normal[2];
            let b = -normal[1] / normal[2];
            let p = points[0];
            depth = Some([a, b, p[2] - a * p[0] - b * p[1]]);
        }
    }
    Primitive {
        points,
        planes,
        depth,
        edge,
        bounds: b,
    }
}
struct Node {
    bounds: Bounds,
    start: usize,
    end: usize,
    children: Option<(usize, usize)>,
}
fn build(nodes: &mut Vec<Node>, ids: &mut [usize], start: usize, p: &[Primitive]) -> usize {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for &i in ids.iter() {
        let a = p[i].bounds;
        for k in 0..2 {
            b[k] = b[k].min(a[k]);
            b[k + 2] = b[k + 2].max(a[k + 2]);
            let c = a[k] * 0.5 + a[k + 2] * 0.5;
            lo[k] = lo[k].min(c);
            hi[k] = hi[k].max(c);
        }
    }
    let idx = nodes.len();
    nodes.push(Node {
        bounds: b,
        start,
        end: start + ids.len(),
        children: None,
    });
    if ids.len() > 32 {
        let axis = usize::from(hi[1] - lo[1] > hi[0] - lo[0]);
        let mid = ids.len() / 2;
        ids.select_nth_unstable_by(mid, |&i, &j| {
            let a = p[i].bounds;
            let b = p[j].bounds;
            (a[axis] * 0.5 + a[axis + 2] * 0.5)
                .total_cmp(&(b[axis] * 0.5 + b[axis + 2] * 0.5))
                .then(i.cmp(&j))
        });
        let (l, r) = ids.split_at_mut(mid);
        let left = build(nodes, l, start, p);
        let right = build(nodes, r, start + mid, p);
        nodes[idx].children = Some((left, right));
    }
    idx
}
fn clip_faces(
    a: &Primitive,
    b: &Primitive,
    eps: f64,
    buf: &mut Vec<P2>,
    tmp: &mut Vec<P2>,
) -> Option<(f64, f64)> {
    let (da, db) = (a.depth?, b.depth?);
    buf.clear();
    buf.extend(a.points.iter().map(|p| [p[0], p[1]]));
    for &plane in &b.planes {
        if buf.is_empty() {
            return None;
        }
        tmp.clear();
        let mut prev = *buf.last().unwrap();
        let mut d0 = plane_value(plane, prev);
        for &cur in buf.iter() {
            let d1 = plane_value(plane, cur);
            if (d0 >= -eps) != (d1 >= -eps) {
                let t = d0 / (d0 - d1);
                tmp.push([
                    prev[0] + t * (cur[0] - prev[0]),
                    prev[1] + t * (cur[1] - prev[1]),
                ]);
            }
            if d1 >= -eps {
                tmp.push(cur);
            }
            prev = cur;
            d0 = d1;
        }
        std::mem::swap(buf, tmp);
    }
    if buf.is_empty() {
        return None;
    }
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for &x in buf.iter() {
        let d = plane_value(da, x) - plane_value(db, x);
        low = low.min(d);
        high = high.max(d);
    }
    Some((low, high))
}
fn line_face(edge: &Primitive, face: &Primitive, eps: f64) -> Option<(f64, f64)> {
    let d = face.depth?;
    let p = edge.points[0];
    let q = edge.points[1];
    let v = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
    let mut low: f64 = 0.;
    let mut high: f64 = 1.;
    for &plane in &face.planes {
        let dist = plane[0] * p[0] + plane[1] * p[1] + plane[2];
        let delta = v[0] * plane[0] + v[1] * plane[1];
        if delta.abs() <= eps {
            if dist < -eps {
                return None;
            }
        } else if delta > eps {
            low = low.max(-dist / delta);
        } else {
            high = high.min(-dist / delta);
        }
    }
    if low > high + eps {
        return None;
    }
    let first = p[2] - (p[0] * d[0] + p[1] * d[1] + d[2]);
    let step = v[2] - (v[0] * d[0] + v[1] * d[1]);
    let a = first + low * step;
    let b = first + high * step;
    Some((a.min(b), a.max(b)))
}
fn lines(a: &Primitive, b: &Primitive, eps: f64, margin: f64) -> Option<(f64, f64)> {
    let (p, q, r, s) = (a.points[0], a.points[1], b.points[0], b.points[1]);
    let (dx, dy, ex, ey) = (q[0] - p[0], q[1] - p[1], s[0] - r[0], s[1] - r[1]);
    let cross = dx * ey - dy * ex;
    if cross.abs() > eps {
        let t = ((r[0] - p[0]) * ey - (r[1] - p[1]) * ex) / cross;
        let u = ((r[0] - p[0]) * dy - (r[1] - p[1]) * dx) / cross;
        if -eps <= t && t <= 1. + eps && -eps <= u && u <= 1. + eps {
            let d = p[2] + t * (q[2] - p[2]) - r[2] - u * (s[2] - r[2]);
            return Some((d, d));
        }
        return None;
    }
    let len = (dx * dx + dy * dy).sqrt();
    if len <= eps || ((r[0] - p[0]) * dy - (r[1] - p[1]) * dx).abs() > margin * len + eps {
        return None;
    }
    let axis = usize::from(dx.abs() < dy.abs());
    let low = p[axis].min(q[axis]).max(r[axis].min(s[axis]));
    let high = p[axis].max(q[axis]).min(r[axis].max(s[axis]));
    if low > high + eps || (s[axis] - r[axis]).abs() <= eps {
        return None;
    }
    let diff = |x: f64| {
        p[2] + (x - p[axis]) / (q[axis] - p[axis]) * (q[2] - p[2])
            - r[2]
            - (x - r[axis]) / (s[axis] - r[axis]) * (s[2] - r[2])
    };
    let (d0, d1) = (diff(low), diff(high));
    Some((d0.min(d1), d0.max(d1)))
}
fn solve(
    p: &[Primitive],
    ranks: &[usize],
    eps: f64,
    de: f64,
    margin: f64,
) -> (Vec<usize>, usize, usize, usize, usize) {
    let n = p.len();
    if n == 0 {
        return (vec![], 0, 0, 0, 0);
    }
    let mut ids: Vec<usize> = (0..n).collect();
    let mut nodes = Vec::new();
    build(&mut nodes, &mut ids, 0, p);
    let mut graph: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    let mut incoming = vec![0usize; n];
    let (mut conflicts, mut candidates, mut constraints) = (0, 0, 0);
    let (mut buf, mut tmp) = (Vec::new(), Vec::new());
    let mut stack = vec![(0, 0)];
    while let Some((i, j)) = stack.pop() {
        let (a, b) = (&nodes[i], &nodes[j]);
        if !overlaps(a.bounds, b.bounds, eps) {
            continue;
        }
        match (a.children, b.children) {
            (Some((l, r)), _) if i == j => {
                stack.extend([(l, l), (l, r), (r, r)]);
            }
            (None, None) => {
                for ia in a.start..a.end {
                    for ib in if i == j { ia + 1 } else { b.start }..b.end {
                        let (i, j) = (ids[ia].min(ids[ib]), ids[ia].max(ids[ib]));
                        let (x, y) = (&p[i], &p[j]);
                        if !overlaps(x.bounds, y.bounds, eps) {
                            continue;
                        }
                        candidates += 1;
                        let diff = match (x.edge, y.edge) {
                            (false, false) => clip_faces(x, y, eps, &mut buf, &mut tmp),
                            (true, true) => lines(x, y, eps, margin),
                            (true, false) => line_face(x, y, eps),
                            (false, true) => line_face(y, x, eps).map(|(l, h)| (-h, -l)),
                        };
                        if let Some((low, high)) = diff {
                            let pair = if high <= de && low < -de {
                                Some((i, j))
                            } else if low >= -de && high > de {
                                Some((j, i))
                            } else {
                                if low < -de && high > de {
                                    conflicts += 1;
                                }
                                None
                            };
                            if let Some((behind, front)) = pair {
                                graph[behind].push(front);
                                incoming[front] += 1;
                                constraints += 1;
                            }
                        }
                    }
                }
            }
            (Some((l, r)), None) => stack.extend([(l, j), (r, j)]),
            (None, Some((l, r))) => stack.extend([(i, l), (i, r)]),
            (Some((l, r)), Some((ll, rr))) => {
                let extent = |a: Bounds| (a[2] - a[0]).max(a[3] - a[1]);
                if extent(a.bounds) >= extent(b.bounds) {
                    stack.extend([(l, j), (r, j)]);
                } else {
                    stack.extend([(i, ll), (i, rr)]);
                }
            }
        }
    }
    let bytes = graph.capacity() * std::mem::size_of::<Vec<usize>>()
        + graph
            .iter()
            .map(|v| v.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
        + incoming.capacity() * std::mem::size_of::<usize>();
    let mut ready = BinaryHeap::new();
    let mut emitted = vec![false; n];
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        if incoming[i] == 0 {
            ready.push(Reverse((ranks[i], i)));
        }
    }
    // The fallback always chooses the smallest still-unemitted stable key.
    let mut by_rank: Vec<usize> = (0..n).collect();
    by_rank.sort_unstable_by_key(|&i| (ranks[i], i));
    let mut cursor = 0;
    while result.len() < n {
        if ready.is_empty() {
            while emitted[by_rank[cursor]] {
                cursor += 1;
            }
            let i = by_rank[cursor];
            ready.push(Reverse((ranks[i], i)));
            conflicts += 1;
        }
        let Reverse((_, i)) = ready.pop().unwrap();
        if emitted[i] {
            continue;
        }
        emitted[i] = true;
        result.push(i);
        for &j in &graph[i] {
            incoming[j] -= 1;
            if incoming[j] == 0 {
                ready.push(Reverse((ranks[j], j)));
            }
        }
    }
    (result, conflicts, candidates, constraints, bytes)
}
#[pyfunction]
#[pyo3(signature=(points, counts, edges, normals, ranks, extent, margin, depth_bias=0.0))]
fn sort_packed(
    py: Python<'_>,
    points: &[u8],
    counts: &[u8],
    edges: &[u8],
    normals: &[u8],
    ranks: &[u8],
    extent: f64,
    margin: f64,
    depth_bias: f64,
) -> PyResult<(Vec<usize>, usize, usize, usize, usize)> {
    let points = read_f64(points)?;
    let counts = read_usize(counts)?;
    let normals = read_f64(normals)?;
    let ranks = read_usize(ranks)?;
    let n = counts.len();
    if !extent.is_finite()
        || extent < 1.
        || !margin.is_finite()
        || !depth_bias.is_finite()
        || edges.len() != n
        || ranks.len() != n
        || normals.len() != n.saturating_mul(3)
        || points.len() % 3 != 0
        || points.iter().chain(normals.iter()).any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err("invalid packed geometry"));
    }
    let mut off = 0usize;
    let mut p = Vec::with_capacity(n);
    let eps = extent * 1e-10;
    for i in 0..n {
        let count = counts[i];
        let edge = edges[i] != 0;
        if (edge && count != 2) || (!edge && count < 3) || count > (points.len() - off) / 3 {
            return Err(PyValueError::new_err("invalid primitive size"));
        }
        let mut row: Vec<P3> = points[off..off + count * 3]
            .chunks_exact(3)
            .map(|x| [x[0], x[1], x[2]])
            .collect();
        off += count * 3;
        if edge && depth_bias != 0. {
            for q in &mut row {
                q[2] += depth_bias;
            }
        }
        p.push(primitive(
            row,
            [normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]],
            edge,
            margin,
            eps,
        ));
    }
    if off != points.len() {
        return Err(PyValueError::new_err("unused coordinates"));
    }
    Ok(py.detach(|| solve(&p, &ranks, eps, extent * 1e-7, margin)))
}
#[pymodule]
fn _ucg_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<render::GlSubmission>()?;
    m.add_class::<resources::GlResources>()?;
    m.add_class::<frame::GlFrame>()?;
    m.add_class::<frame::GlIndexedBlend>()?;
    m.add_function(wrap_pyfunction!(sort_packed, m)?)?;
    m.add_function(wrap_pyfunction!(contacts::contact_arcs, m)?)?;
    Ok(())
}
