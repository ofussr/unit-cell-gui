//! Context-local frame parameters and WBOIT state transactions.
use pyo3::{prelude::*, exceptions::{PyValueError,PyRuntimeError}};
use std::ffi::c_void;
const UBO:u32=0x8a11;
const UBO_BINDING:u32=0x8a28;
const SLOT:u32=7;
const TEX0:u32=0x84c0;
const STATE:[u32;13]=[0x8ca6,0x8caa,0x8b8d,0x85b5,0x84e0,0x8ca7,0x0b74,0x80c9,0x80c8,0x80cb,0x80ca,0x8009,0x883d];
const CAPS:[u32;7]=[0x0be2,0x0b71,0x809d,0x0c11,0x8db9,0x0b44,0x809e];
const TARGETS:[(u32,u32);3]=[(0x0de1,0x8069),(0x9100,0x9104),(0x8c2a,0x8c2c)];
#[pyclass(unsendable,module="unit_cell_gui._ucg_native")]
pub struct GlFrame {
    get_i:unsafe extern "system" fn(u32,*mut i32),
    get_ii:unsafe extern "system" fn(u32,u32,*mut i32),
    get_i64:unsafe extern "system" fn(u32,u32,*mut i64),
    get_b:unsafe extern "system" fn(u32,*mut u8),
    is_enabled:unsafe extern "system" fn(u32)->u8,
    use_program:unsafe extern "system" fn(u32),
    uniform_i:unsafe extern "system" fn(i32,i32),
    uniform_f:unsafe extern "system" fn(i32,f32),
    bind_buffer:unsafe extern "system" fn(u32,u32),
    buffer_sub:unsafe extern "system" fn(u32,isize,isize,*const c_void),
    bind_base:unsafe extern "system" fn(u32,u32,u32),
    bind_range:unsafe extern "system" fn(u32,u32,u32,isize,isize),
    active:unsafe extern "system" fn(u32),
    bind_tex:unsafe extern "system" fn(u32,u32),
    bind_fbo:unsafe extern "system" fn(u32,u32),
    bind_vao:unsafe extern "system" fn(u32),
    bind_rbo:unsafe extern "system" fn(u32,u32),
    viewport:unsafe extern "system" fn(i32,i32,i32,i32),
    depth_func:unsafe extern "system" fn(u32),
    depth_mask:unsafe extern "system" fn(u8),
    color_mask:unsafe extern "system" fn(u8,u8,u8,u8),
    blend_func:unsafe extern "system" fn(u32,u32,u32,u32),
    blend_eq:unsafe extern "system" fn(u32,u32),
    enable:unsafe extern "system" fn(u32),
    disable:unsafe extern "system" fn(u32),
    get_error:unsafe extern "system" fn()->u32,
}
#[pymethods]
impl GlFrame {
    #[new]
    fn new(a:Vec<usize>)->PyResult<Self>{
        if a.len()!=26||a.iter().any(|&v|v<4096||v>=usize::MAX-2){return Err(PyValueError::new_err("missing frame GL entry point"));}
        macro_rules! f {($i:expr)=>{std::mem::transmute(a[$i])}}
        unsafe{Ok(Self{get_i:f!(0),get_ii:f!(1),get_i64:f!(2),get_b:f!(3),is_enabled:f!(4),use_program:f!(5),uniform_i:f!(6),uniform_f:f!(7),bind_buffer:f!(8),buffer_sub:f!(9),bind_base:f!(10),bind_range:f!(11),active:f!(12),bind_tex:f!(13),bind_fbo:f!(14),bind_vao:f!(15),bind_rbo:f!(16),viewport:f!(17),depth_func:f!(18),depth_mask:f!(19),color_mask:f!(20),blend_func:f!(21),blend_eq:f!(22),enable:f!(23),disable:f!(24),get_error:f!(25)})}
    }
    /// Capture generic and indexed UBO bindings (including an existing range).
    fn bind_frame(&self,buffer:u32,data:Option<Vec<f32>>)->PyResult<(u32,u32,i64,i64)>{
        if buffer==0 || data.as_ref().is_some_and(|v|v.len()!=36 || v.iter().any(|x|!x.is_finite())) {return Err(PyValueError::new_err("invalid frame buffer/data"));}
        let mut generic=0;let mut indexed=0;let mut start=0;let mut size=0;
        unsafe{
            (self.get_i)(UBO_BINDING,&mut generic);
            (self.get_ii)(UBO_BINDING,SLOT,&mut indexed);
            (self.get_i64)(0x8a29,SLOT,&mut start);
            (self.get_i64)(0x8a2a,SLOT,&mut size);
            (self.bind_base)(UBO,SLOT,buffer);
            if let Some(v)=data {(self.buffer_sub)(UBO,0,144,v.as_ptr().cast());}
        }
        let state=(generic as u32,indexed as u32,start,size);
        if let Err(e)=self.check(){self.restore_frame(state)?;return Err(e);}
        Ok(state)
    }
    fn restore_frame(&self,state:(u32,u32,i64,i64))->PyResult<()>{
        let (generic,indexed,start,size)=state;
        if start<0||size<0||start>isize::MAX as i64||size>isize::MAX as i64{return Err(PyValueError::new_err("invalid UBO range"));}
        unsafe{
            if indexed!=0&&size>0 {(self.bind_range)(UBO,SLOT,indexed,start as isize,size as isize);}
            else {(self.bind_base)(UBO,SLOT,indexed);}
            (self.bind_buffer)(UBO,generic);
        }
        self.check()
    }
    fn uniforms(&self,program:u32,ints:Vec<(i32,i32)>,floats:Vec<(i32,f32)>)->PyResult<()>{
        if program==0||floats.iter().any(|(_,v)|!v.is_finite()){return Err(PyValueError::new_err("invalid uniform batch"));}
        unsafe{(self.use_program)(program);for (l,v) in ints {if l>=0{(self.uniform_i)(l,v);}}for(l,v)in floats{if l>=0{(self.uniform_f)(l,v);}}}
        self.check()
    }
    /// Snapshot exactly the same state as the legacy WBOIT guard.
    fn save_state(&self)->PyResult<Vec<i32>>{
        let mut s=vec![0;41];
        unsafe{
            for (i,p) in STATE.iter().enumerate(){(self.get_i)(*p,&mut s[i]);}
            (self.get_i)(0x0ba2,s[13..17].as_mut_ptr());
            let mut depth=0u8;let mut color=[0u8;4];
            (self.get_b)(0x0b72,&mut depth);(self.get_b)(0x0c23,color.as_mut_ptr());
            s[17]=depth as i32;for i in 0..4{s[18+i]=color[i] as i32;}
            for(i,p)in CAPS.iter().enumerate(){s[22+i]=(self.is_enabled)(*p) as i32;}
            for unit in 0..4{(self.active)(TEX0+unit as u32);for (j,(_,query)) in TARGETS.iter().enumerate(){(self.get_i)(*query,&mut s[29+3*unit+j]);}}
            (self.active)(s[4] as u32);
        }
        self.check()?;Ok(s)
    }
    fn restore_state(&self,s:Vec<i32>)->PyResult<()>{
        if s.len()!=41{return Err(PyValueError::new_err("invalid OpenGL snapshot"));}
        unsafe{
            (self.bind_fbo)(0x8ca9,s[0] as u32);(self.bind_fbo)(0x8ca8,s[1] as u32);
            (self.viewport)(s[13],s[14],s[15],s[16]);(self.use_program)(s[2] as u32);(self.bind_vao)(s[3] as u32);
            (self.bind_rbo)(0x8d41,s[5] as u32);(self.depth_func)(s[6] as u32);(self.depth_mask)(s[17] as u8);
            (self.color_mask)(s[18] as u8,s[19] as u8,s[20] as u8,s[21] as u8);
            (self.blend_func)(s[7] as u32,s[8] as u32,s[9] as u32,s[10] as u32);(self.blend_eq)(s[11] as u32,s[12] as u32);
            for (i,p) in CAPS.iter().enumerate(){if s[22+i]!=0{(self.enable)(*p);}else{(self.disable)(*p);}}
            for unit in 0..4{(self.active)(TEX0+unit as u32);for(j,(target,_))in TARGETS.iter().enumerate(){(self.bind_tex)(*target,s[29+3*unit+j] as u32);}}
            (self.active)(s[4] as u32);
        }
        self.check()
    }
    /// State of a blended line/face pass; only called at explicit pass boundaries.
    fn blend_pass(&self,enable:bool)->PyResult<()>{
        unsafe{if enable{(self.enable)(0x0be2);(self.blend_func)(0x0302,0x0303,1,0x0303);(self.depth_mask)(0);}else{(self.depth_mask)(1);(self.disable)(0x0be2);}}
        self.check()
    }
}
impl GlFrame{fn check(&self)->PyResult<()>{let e=unsafe{(self.get_error)()};if e==0{Ok(())}else{Err(PyRuntimeError::new_err(format!("frame OpenGL error 0x{e:04x}")))}}}

/// Optional independent blend state. Only constructed on GL 4.0 or
/// ARB_draw_buffers_blend contexts, with the correct core/ARB entry points.
#[pyclass(unsendable,module="unit_cell_gui._ucg_native")]
pub struct GlIndexedBlend {
    get_i:unsafe extern "system" fn(u32,*mut i32),
    get_ii:unsafe extern "system" fn(u32,u32,*mut i32),
    get_bi:unsafe extern "system" fn(u32,u32,*mut u8),
    is_enabled:unsafe extern "system" fn(u32,u32)->u8,
    enable:unsafe extern "system" fn(u32,u32),
    disable:unsafe extern "system" fn(u32,u32),
    blend:unsafe extern "system" fn(u32,u32,u32,u32,u32),
    equation:unsafe extern "system" fn(u32,u32,u32),
    mask:unsafe extern "system" fn(u32,u8,u8,u8,u8),
    get_error:unsafe extern "system" fn()->u32,
    count:u32,
}
#[pymethods]
impl GlIndexedBlend {
    #[new]
    fn new(a:Vec<usize>)->PyResult<Self>{
        if a.len()!=10||a.iter().any(|&v|v<4096||v>=usize::MAX-2){return Err(PyValueError::new_err("missing independent blend GL entry point"));}
        macro_rules! f {($i:expr)=>{std::mem::transmute(a[$i])}}
        let mut result=unsafe{Self{get_i:f!(0),get_ii:f!(1),get_bi:f!(2),is_enabled:f!(3),enable:f!(4),disable:f!(5),blend:f!(6),equation:f!(7),mask:f!(8),get_error:f!(9),count:0}};
        let mut count=0;
        unsafe{(result.get_i)(0x8824,&mut count);}
        result.check()?;
        if !(2..=256).contains(&count){return Err(PyRuntimeError::new_err("independent blending requires at least two draw buffers"));}
        result.count=count as u32;
        Ok(result)
    }
    fn save(&self)->PyResult<Vec<[i32;11]>>{
        let mut state=Vec::with_capacity(self.count as usize);
        unsafe{for i in 0..self.count{
            let mut row=[0;11];
            for (j,p) in [0x80c9,0x80c8,0x80cb,0x80ca,0x8009,0x883d].iter().enumerate(){(self.get_ii)(*p,i,&mut row[j]);}
            row[6]=(self.is_enabled)(0x0be2,i) as i32;
            let mut mask=[0u8;4];(self.get_bi)(0x0c23,i,mask.as_mut_ptr());
            for j in 0..4{row[7+j]=mask[j] as i32;}
            state.push(row);
        }}
        self.check()?;Ok(state)
    }
    fn restore(&self,state:Vec<[i32;11]>)->PyResult<()>{
        if state.len()!=self.count as usize{return Err(PyValueError::new_err("invalid indexed blend snapshot"));}
        unsafe{for (i,row) in state.iter().enumerate(){let i=i as u32;
            (self.blend)(i,row[0] as u32,row[1] as u32,row[2] as u32,row[3] as u32);
            (self.equation)(i,row[4] as u32,row[5] as u32);
            if row[6]!=0{(self.enable)(0x0be2,i);}else{(self.disable)(0x0be2,i);}
            (self.mask)(i,row[7] as u8,row[8] as u8,row[9] as u8,row[10] as u8);
        }}
        self.check()
    }
    fn configure(&self)->PyResult<()>{
        unsafe{
            for i in 0..2{(self.enable)(0x0be2,i);(self.equation)(i,0x8006,0x8006);(self.mask)(i,1,1,1,1);}
            (self.blend)(0,1,1,1,1);
            (self.blend)(1,0,0x0301,0,0x0301);
        }
        self.check()
    }
}
impl GlIndexedBlend{fn check(&self)->PyResult<()>{let e=unsafe{(self.get_error)()};if e==0{Ok(())}else{Err(PyRuntimeError::new_err(format!("indexed blend OpenGL error 0x{e:04x}")))}}}
