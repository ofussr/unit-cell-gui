//! Context-local OpenGL submission. Qt owns context and all GL resources.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

type Descriptor = (u32, u32, i32, u32, Vec<(u32, i32)>); // program, vao, uniform, mode, (texture, size)
type Range = (usize, i32, i32); // descriptor, start, count
const TEXTURE0: u32 = 0x84c0;
const TEXTURE_BUFFER: u32 = 0x8c2a;
const LINES: u32 = 1;
const TRIANGLES: u32 = 4;

#[pyclass(unsendable, module = "unit_cell_gui._ucg_native")]
pub struct GlSubmission {
    use_program: unsafe extern "system" fn(u32),
    bind_vao: unsafe extern "system" fn(u32),
    active_texture: unsafe extern "system" fn(u32),
    bind_texture: unsafe extern "system" fn(u32, u32),
    uniform_i: unsafe extern "system" fn(i32, i32),
    draw_instanced: unsafe extern "system" fn(u32, i32, i32, i32),
    get_error: unsafe extern "system" fn() -> u32,
    uniform_f: unsafe extern "system" fn(i32, f32),
    draw_arrays: unsafe extern "system" fn(u32, i32, i32),
    draw_elements: unsafe extern "system" fn(u32, i32, u32, *const std::ffi::c_void, i32),
    attrib_i: unsafe extern "system" fn(u32, i32, u32, i32, *const std::ffi::c_void),
}

#[pymethods]
impl GlSubmission {
    #[new]
    fn new(addresses: Vec<usize>) -> PyResult<Self> {
        if addresses.len() != 11 || addresses.iter().any(|&a| a < 4096 || a >= usize::MAX - 2) {
            return Err(PyValueError::new_err("missing OpenGL entry point"));
        }
        // Addresses are supplied by the current QOpenGLContext. Never shared
        // across contexts/threads or used after widget cleanup.
        unsafe {
            Ok(Self {
                use_program: std::mem::transmute::<usize, unsafe extern "system" fn(u32)>(addresses[0]),
                bind_vao: std::mem::transmute::<usize, unsafe extern "system" fn(u32)>(addresses[1]),
                active_texture: std::mem::transmute::<usize, unsafe extern "system" fn(u32)>(addresses[2]),
                bind_texture: std::mem::transmute::<usize, unsafe extern "system" fn(u32,u32)>(addresses[3]),
                uniform_i: std::mem::transmute::<usize, unsafe extern "system" fn(i32,i32)>(addresses[4]),
                draw_instanced: std::mem::transmute::<usize, unsafe extern "system" fn(u32,i32,i32,i32)>(addresses[5]),
                get_error: std::mem::transmute::<usize, unsafe extern "system" fn()->u32>(addresses[6]),
                uniform_f: std::mem::transmute::<usize, unsafe extern "system" fn(i32,f32)>(addresses[7]),
                draw_arrays: std::mem::transmute::<usize, unsafe extern "system" fn(u32,i32,i32)>(addresses[8]),
                draw_elements: std::mem::transmute::<usize, unsafe extern "system" fn(u32,i32,u32,*const std::ffi::c_void,i32)>(addresses[9]),
                attrib_i: std::mem::transmute::<usize, unsafe extern "system" fn(u32,i32,u32,i32,*const std::ffi::c_void)>(addresses[10]),
            })
        }
    }

    /// Validate and expand all ranges before issuing any GL call. Consecutive
    /// ranges can be merged but no primitive may move in the draw sequence.
    fn draw_ranges(&self, descriptors: Vec<Descriptor>, ranges: Vec<Range>, merge: bool) -> PyResult<usize> {
        let totals: Vec<i64> = descriptors.iter().map(|(program, vao, _, mode, chunks)| {
            if *program == 0 || *vao == 0 || !matches!(*mode, LINES | TRIANGLES) || chunks.is_empty()
                || chunks.iter().any(|&(texture,size)| texture == 0 || size < 0) {
                return Err(PyValueError::new_err("invalid OpenGL descriptor"));
            }
            Ok(chunks.iter().map(|&(_,size)| i64::from(size)).sum())
        }).collect::<PyResult<_>>()?;
        let mut runs: Vec<Range> = Vec::with_capacity(ranges.len());
        for (id, first, count) in ranges {
            if id >= totals.len() || first < 0 || count < 0 || i64::from(first)+i64::from(count)>totals[id] {
                return Err(PyValueError::new_err("draw range outside resident buffer"));
            }
            if count == 0 { continue; }
            if merge {
                if let Some(last) = runs.last_mut() {
                    if last.0 == id && last.1.checked_add(last.2) == Some(first) {
                        if let Some(n) = last.2.checked_add(count) { last.2=n; continue; }
                    }
                }
            }
            runs.push((id,first,count));
        }
        let mut draws=0;
        let mut bound=None;
        let mut texture=0;
        unsafe {
            (self.active_texture)(TEXTURE0);
            for (id,mut first,mut count) in runs {
                let (program,vao,location,mode,chunks)=&descriptors[id];
                if bound != Some(id) {
                    (self.use_program)(*program);
                    (self.bind_vao)(*vao);
                    bound=Some(id);
                }
                for &(tex,size) in chunks {
                    if first >= size { first-=size; continue; }
                    let n=count.min(size-first);
                    if tex!=texture { (self.bind_texture)(TEXTURE_BUFFER,tex); texture=tex; }
                    (self.uniform_i)(*location,first);
                    (self.draw_instanced)(*mode,0,if *mode==TRIANGLES {3} else {2},n);
                    draws+=1;
                    count-=n;
                    if count==0 {break;}
                    first=0;
                }
            }
            (self.bind_vao)(0);
        }
        self.check_error()?;
        Ok(draws)
    }

    /// Draw arrays or instanced arrays from already configured VAOs.
    /// A negative-one instance count denotes a non-instanced draw.
    fn draw_arrays_batch(&self, records: Vec<(u32,u32,u32,i32,i32,i32)>) -> PyResult<usize> {
        if records.iter().any(|&(vao,_,mode,first,count,instances)| vao==0 || !matches!(mode,1|4|5)
            || first<0 || count<0 || instances < -1) {
            return Err(PyValueError::new_err("invalid array draw"));
        }
        let mut draws=0;
        let mut last_vao=0;
        let mut last_texture=0;
        unsafe {
            for (vao,texture,mode,first,count,instances) in records {
                if count==0 || instances==0 {continue;}
                if texture!=0 && texture!=last_texture {
                    (self.active_texture)(TEXTURE0);
                    (self.bind_texture)(TEXTURE_BUFFER,texture); last_texture=texture;
                }
                if vao!=last_vao {(self.bind_vao)(vao);last_vao=vao;}
                if instances==-1 {(self.draw_arrays)(mode,first,count);}
                else {(self.draw_instanced)(mode,first,count,instances);}
                draws+=1;
            }
            (self.bind_vao)(0);
        }
        self.check_error()?;
        Ok(draws)
    }

    fn draw_indexed(&self, vao:u32, count:i32, instances:i32) -> PyResult<usize> {
        if vao==0 || count<0 || instances<0 {return Err(PyValueError::new_err("invalid indexed draw"));}
        if count==0 || instances==0 {return Ok(0);}
        unsafe {
            (self.bind_vao)(vao);
            (self.draw_elements)(TRIANGLES,count,0x1405,std::ptr::null(),instances);
            (self.bind_vao)(0);
        }
        self.check_error()?;
        Ok(1)
    }

    /// Indices are in the resident edge-order VBO; preserve their sequence.
    fn draw_ordered_edges(&self, location:i32, chunks:Vec<(u32,i32)>, indices:Vec<i32>) -> PyResult<usize> {
        if indices.is_empty() {return Ok(0);}
        if chunks.is_empty() || chunks[0].1<=0 || chunks.iter().any(|&(t,n)| t==0 || n<=0)
            || indices.len()>i32::MAX as usize {
            return Err(PyValueError::new_err("invalid ordered edge buffers"));
        }
        let capacity=chunks[0].1;
        if chunks[..chunks.len()-1].iter().any(|&(_,n)| n!=capacity) || chunks.last().unwrap().1>capacity
            || indices.iter().any(|&i| i<0 || (i/capacity) as usize>=chunks.len()
            || i%capacity>=chunks[(i/capacity) as usize].1) {
            return Err(PyValueError::new_err("invalid ordered edge index"));
        }
        let mut draws=0;
        let mut first=0;
        unsafe {
            while first<indices.len() {
                let chunk=indices[first]/capacity;
                let mut end=first+1;
                while end<indices.len() && indices[end]/capacity==chunk {end+=1;}
                (self.bind_texture)(TEXTURE_BUFFER,chunks[chunk as usize].0);
                (self.uniform_i)(location,chunk*capacity);
                (self.attrib_i)(1,1,0x1404,4,(first*4) as *const std::ffi::c_void);
                (self.draw_instanced)(LINES,0,2,(end-first) as i32);
                draws+=1;first=end;
            }
        }
        self.check_error()?;
        Ok(draws)
    }

    /// Ordinary line ranges (atom/face intersection ink), preserving opacity.
    fn draw_sections(&self, opacity_location:i32, ranges:Vec<(u32,i32,i32,f32)>) -> PyResult<usize> {
        if ranges.iter().any(|&(vao,first,count,alpha)| vao==0 || first<0 || count<0 || !alpha.is_finite()) {
            return Err(PyValueError::new_err("invalid section range"));
        }
        let mut draws=0;
        let mut vao=0;
        let mut opacity=None;
        unsafe {
            for (v,first,count,alpha) in ranges {
                if count==0 {continue;}
                if opacity!=Some(alpha) {(self.uniform_f)(opacity_location,alpha);opacity=Some(alpha);}
                if vao!=v {(self.bind_vao)(v);vao=v;}
                (self.draw_arrays)(LINES,first,count); draws+=1;
            }
            (self.bind_vao)(0);
        }
        self.check_error()?;
        Ok(draws)
    }
}
impl GlSubmission {
    fn check_error(&self)->PyResult<()> {
        let error=unsafe{(self.get_error)()};
        if error!=0 {Err(PyRuntimeError::new_err(format!("native OpenGL submission error 0x{error:04x}")))} else {Ok(())}
    }
}
