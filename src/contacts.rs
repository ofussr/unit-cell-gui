//! Clip the actual circle to convex face half-planes, then tessellate only
//! accepted arcs at the original 160-segment angular resolution.
use super::{P3, read_f64};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use std::f64::consts::TAU;
const N: usize = 160;
fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn point(c: P3, u: P3, v: P3, r: f64, cs: f64, sn: f64) -> P3 {
    std::array::from_fn(|i| c[i] + r * (cs * u[i] + sn * v[i]))
}
fn triples(a: &[f64]) -> Vec<P3> {
    a.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()
}
fn emit(out: &mut Vec<u8>, p: P3) {
    for x in p {
        out.extend_from_slice(&x.to_le_bytes());
    }
}
struct Face<'a> {
    polygon: &'a [P3],
    sides: &'a [P3],
    normal: P3,
    u: P3,
    v: P3,
    sign: f64,
    grid: &'a [f64],
}
impl Face<'_> {
    fn grid_point(&self, c: P3, r: f64, i: usize) -> P3 {
        let i = i % N;
        point(c, self.u, self.v, r, self.grid[2 * i], self.grid[2 * i + 1])
    }
    fn angle_point(&self, c: P3, r: f64, t: f64) -> P3 {
        let (s, k) = t.sin_cos();
        point(c, self.u, self.v, r, k, s)
    }
    fn arc(&self, out: &mut Vec<u8>, c: P3, r: f64, lo: f64, hi: f64) {
        let step = TAU / N as f64;
        let mut previous = if lo == 0. {
            self.grid_point(c, r, 0)
        } else {
            self.angle_point(c, r, lo)
        };
        let mut k = (lo / step).floor() as usize + 1;
        while k < N && (k as f64) * step < hi - 1e-12 {
            if (k as f64) * step > lo + 1e-12 {
                let next = self.grid_point(c, r, k);
                emit(out, previous);
                emit(out, next);
                previous = next;
            }
            k += 1;
        }
        let end = if hi == TAU {
            self.grid_point(c, r, 0)
        } else {
            self.angle_point(c, r, hi)
        };
        emit(out, previous);
        emit(out, end);
    }
    fn circle(&self, out: &mut Vec<u8>, atom: P3, radius: f64, signed: f64) {
        let square = radius * radius - signed * signed;
        if square <= 1e-12 {
            return;
        }
        let r = square.sqrt();
        let c = std::array::from_fn(|i| atom[i] - signed * self.normal[i]);
        let mut planes = Vec::with_capacity(self.polygon.len());
        let mut cuts = vec![0., TAU];
        for (&p, &s) in self.polygon.iter().zip(self.sides) {
            // sign is chosen from a point inside the convex polygon, making
            // winding and outward normal independent.
            let a = self.sign * r * dot(self.u, s);
            let b = self.sign * r * dot(self.v, s);
            let d = self.sign * dot(sub(c, p), s);
            let amp = a.hypot(b);
            if d >= amp {
                continue;
            }
            if d <= -amp {
                return;
            }
            let phase = b.atan2(a);
            let angle = (-d / amp).clamp(-1., 1.).acos();
            cuts.push((phase - angle).rem_euclid(TAU));
            cuts.push((phase + angle).rem_euclid(TAU));
            planes.push([a, b, d]);
        }
        if planes.is_empty() {
            for i in 0..N {
                emit(out, self.grid_point(c, r, i));
                emit(out, self.grid_point(c, r, i + 1));
            }
            return;
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let mut start = None;
        for w in cuts.windows(2) {
            let (s, k) = ((w[0] + w[1]) * 0.5).sin_cos();
            let inside = planes.iter().all(|p| p[0] * k + p[1] * s + p[2] >= 0.);
            if inside {
                if start.is_none() {
                    start = Some(w[0]);
                }
            } else if let Some(lo) = start.take() {
                self.arc(out, c, r, lo, w[0]);
            }
        }
        if let Some(lo) = start {
            self.arc(out, c, r, lo, *cuts.last().unwrap());
        }
    }
}
#[pyfunction]
pub fn contact_arcs<'py>(
    py: Python<'py>,
    polygon: &[u8],
    sides: &[u8],
    basis: &[u8],
    atoms: &[u8],
    grid: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let polygon = read_f64(polygon)?;
    let sides = read_f64(sides)?;
    let basis = read_f64(basis)?;
    let atoms = read_f64(atoms)?;
    let grid = read_f64(grid)?;
    if polygon.len() < 9
        || polygon.len() % 3 != 0
        || sides.len() != polygon.len()
        || basis.len() != 9
        || atoms.len() % 5 != 0
        || grid.len() != 2 * N
        || polygon
            .iter()
            .chain(&sides)
            .chain(&basis)
            .chain(&atoms)
            .chain(&grid)
            .any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err("invalid contact geometry"));
    }
    let p = triples(&polygon);
    let s = triples(&sides);
    let b = triples(&basis);
    // Subtract the origin before averaging, for translated structures.
    let centre: P3 =
        std::array::from_fn(|i| p.iter().map(|q| q[i] - p[0][i]).sum::<f64>() / p.len() as f64);
    let sign_test = p
        .iter()
        .zip(&s)
        .map(|(&q, &n)| dot(sub(centre, sub(q, p[0])), n))
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .unwrap_or(0.);
    let face = Face {
        polygon: &p,
        sides: &s,
        normal: b[0],
        u: b[1],
        v: b[2],
        sign: if sign_test < 0. { -1. } else { 1. },
        grid: &grid,
    };
    let out = py.detach(|| {
        let mut out = Vec::new();
        for a in atoms.chunks_exact(5) {
            face.circle(&mut out, [a[0], a[1], a[2]], a[3], a[4]);
        }
        out
    });
    Ok(PyBytes::new(py, &out))
}
