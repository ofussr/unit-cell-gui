//! Context-local resource operations. Qt owns context; Python tracks names.
use pyo3::{prelude::*, exceptions::{PyValueError, PyRuntimeError}, types::PyBytes};
use std::ffi::c_void;
type Gen = unsafe extern "system" fn(i32,*mut u32);
type Delete = unsafe extern "system" fn(i32,*const u32);
type Bind = unsafe extern "system" fn(u32,u32);
type Toggle = unsafe extern "system" fn(u32);
// CPython's public Py_buffer layout is stable from 3.11. Resolve the two
// API functions at runtime to retain cp310-abi3 compatibility. 3.10 uses
// the existing owned bytes path. No raw ndarray address is supplied by Python.
#[repr(C)]
struct BufferView {
    buf:*mut c_void, obj:*mut pyo3::ffi::PyObject, len:isize, itemsize:isize,
    readonly:i32, ndim:i32, format:*mut std::ffi::c_char,
    shape:*mut isize, strides:*mut isize, suboffsets:*mut isize, internal:*mut c_void,
}
type GetBuffer=unsafe extern "C" fn(*mut pyo3::ffi::PyObject,*mut BufferView,i32)->i32;
type ReleaseBuffer=unsafe extern "C" fn(*mut BufferView);
struct BorrowedBuffer<'py> {
    view:Box<BufferView>, release:ReleaseBuffer, _py:Python<'py>,
}
impl Drop for BorrowedBuffer<'_> {
    fn drop(&mut self){unsafe{(self.release)(&mut *self.view)}}
}
enum Payload<'py> { Bytes(Bound<'py,PyBytes>), Buffer(BorrowedBuffer<'py>) }
impl Payload<'_> {
    fn ptr(&self)->*const c_void {match self {Self::Bytes(b)=>b.as_bytes().as_ptr() as *const c_void,Self::Buffer(b)=>b.view.buf}}
    fn len(&self)->isize {match self {Self::Bytes(b)=>b.as_bytes().len() as isize,Self::Buffer(b)=>b.view.len}}
}
#[pyclass(unsendable, module="unit_cell_gui._ucg_native")]
pub struct GlResources {
    generate_fn: [Gen;5], delete: [Delete;5], bind_buffer: Bind, bind_texture: Bind,
    bind_vao: Toggle, enable_attrib: Toggle, disable_attrib: Toggle,
    attrib: unsafe extern "system" fn(u32,i32,u32,u8,i32,*const c_void),
    attrib_i: unsafe extern "system" fn(u32,i32,u32,i32,*const c_void),
    divisor: Bind, tex_buffer: unsafe extern "system" fn(u32,u32,u32),
    data: unsafe extern "system" fn(u32,isize,*const c_void,u32),
    sub: unsafe extern "system" fn(u32,isize,isize,*const c_void),
    error: unsafe extern "system" fn()->u32,
    bind_rb: Bind, bind_fb: Bind,
    rb_storage: unsafe extern "system" fn(u32,u32,i32,i32),
    rb_ms: unsafe extern "system" fn(u32,i32,u32,i32,i32),
    rb_query: unsafe extern "system" fn(u32,u32,*mut i32),
    tex_image: unsafe extern "system" fn(u32,i32,i32,i32,i32,i32,u32,u32,*const c_void),
    tex_ms: unsafe extern "system" fn(u32,i32,u32,i32,i32,u8),
    tex_param: unsafe extern "system" fn(u32,u32,i32),
    fb_tex: unsafe extern "system" fn(u32,u32,u32,u32,i32),
    fb_rb: unsafe extern "system" fn(u32,u32,u32,u32),
    draw_buffer: Toggle, read_buffer: Toggle,
    fb_status: unsafe extern "system" fn(u32)->u32,
    buffer_api: Option<(GetBuffer,ReleaseBuffer)>,
}
impl GlResources {
    fn payload<'py>(&self,obj:&Bound<'py,PyAny>)->PyResult<Payload<'py>> {
        if let Ok(b)=obj.cast::<PyBytes>() {return Ok(Payload::Bytes(b.clone()))}
        let (get,release)=self.buffer_api.ok_or_else(||PyValueError::new_err("buffer requires owned bytes on Python 3.10"))?;
        let py=obj.py();
        // PyBUF_SIMPLE requests a contiguous byte buffer. CPython holds a
        // reference to the exporter until release. Keep the view address fixed.
        let mut view:Box<BufferView>=Box::new(unsafe{std::mem::zeroed()});
        if unsafe{get(obj.as_ptr(),&mut *view,0)}!=0{return Err(PyErr::fetch(py))}
        let borrowed=BorrowedBuffer{view,release,_py:py};
        if borrowed.view.len<0 || (borrowed.view.len>0 && borrowed.view.buf.is_null()) {return Err(PyValueError::new_err("invalid exported buffer"))}
        Ok(Payload::Buffer(borrowed))
    }
    fn check(&self)->PyResult<()> {
        let error=unsafe{(self.error)()};
        if error!=0 {Err(PyRuntimeError::new_err(format!("resource OpenGL error 0x{error:04x}")))} else {Ok(())}
    }
}
#[pymethods]
impl GlResources {
    #[new]
    #[pyo3(signature=(a,buffer_api=None))]
    fn new(a:Vec<usize>,buffer_api:Option<(usize,usize)>)->PyResult<Self> {
        if a.len()!=35 || a.iter().any(|&v|v<4096||v>=usize::MAX-2) {return Err(PyValueError::new_err("missing resource OpenGL entry point"));}
        if buffer_api.is_some_and(|(a,b)|a<4096||b<4096||a>=usize::MAX-2||b>=usize::MAX-2){return Err(PyValueError::new_err("invalid buffer API"))}
        unsafe {
            let buffer_api=buffer_api.map(|(a,b)|(std::mem::transmute::<usize,GetBuffer>(a),std::mem::transmute::<usize,ReleaseBuffer>(b)));
            macro_rules! f {($i:expr)=>{std::mem::transmute(a[$i])}}
            Ok(Self{generate_fn:[f!(0),f!(1),f!(2),f!(3),f!(4)],delete:[f!(5),f!(6),f!(7),f!(8),f!(9)],
                bind_buffer:f!(10),bind_texture:f!(11),bind_vao:f!(12),enable_attrib:f!(13),disable_attrib:f!(14),attrib:f!(15),attrib_i:f!(16),divisor:f!(17),tex_buffer:f!(18),data:f!(19),sub:f!(20),error:f!(21), bind_rb:f!(22),bind_fb:f!(23),rb_storage:f!(24),rb_ms:f!(25),rb_query:f!(26),
                tex_image:f!(27),tex_ms:f!(28),tex_param:f!(29),fb_tex:f!(30),fb_rb:f!(31),draw_buffer:f!(32),read_buffer:f!(33),fb_status:f!(34),buffer_api})
        }
    }
    fn generate(&self,kind:usize)->PyResult<u32> {
        let f=self.generate_fn.get(kind).ok_or_else(||PyValueError::new_err("invalid resource kind"))?;
        let mut id=0;unsafe{f(1,&mut id)};
        if let Err(e)=self.check() {if id!=0 {unsafe{(self.delete[kind])(1,&id)}} return Err(e)}
        if id==0{return Err(PyRuntimeError::new_err("OpenGL returned zero resource name"))} Ok(id)
    }
    fn delete(&self,kind:usize,ids:Vec<u32>)->PyResult<()> {
        let f=self.delete.get(kind).ok_or_else(||PyValueError::new_err("invalid resource kind"))?;
        let n=i32::try_from(ids.len()).map_err(|_|PyValueError::new_err("too many resource names"))?;
        unsafe{f(n,ids.as_ptr())};self.check()
    }
    fn bind_buffer(&self,target:u32,id:u32)->PyResult<()> {unsafe{(self.bind_buffer)(target,id)};self.check()}
    fn bind_texture(&self,target:u32,id:u32)->PyResult<()> {unsafe{(self.bind_texture)(target,id)};self.check()}
    fn bind_vao(&self,id:u32)->PyResult<()> {unsafe{(self.bind_vao)(id)};self.check()}
    fn enable_attrib(&self,id:u32,enabled:bool)->PyResult<()> {unsafe{if enabled{(self.enable_attrib)(id)}else{(self.disable_attrib)(id)}};self.check()}
    fn attrib(&self,id:u32,size:i32,ty:u32,normalized:bool,stride:i32,offset:usize)->PyResult<()> {
        unsafe{(self.attrib)(id,size,ty,normalized as u8,stride,offset as *const c_void)};self.check()
    }
    fn attrib_i(&self,id:u32,size:i32,ty:u32,stride:i32,offset:usize)->PyResult<()> {
        unsafe{(self.attrib_i)(id,size,ty,stride,offset as *const c_void)};self.check()
    }
    fn divisor(&self,id:u32,divisor:u32)->PyResult<()> {unsafe{(self.divisor)(id,divisor)};self.check()}
    fn tex_buffer(&self,target:u32,format:u32,id:u32)->PyResult<()> {unsafe{(self.tex_buffer)(target,format,id)};self.check()}
    #[pyo3(signature=(target,size,usage,payload=None))]
    fn buffer_data(&self,target:u32,size:isize,usage:u32,payload:Option<&Bound<'_,PyAny>>)->PyResult<()> {
        let payload=payload.map(|p|self.payload(p)).transpose()?;
        if size<0 || payload.as_ref().is_some_and(|p|p.len()!=size){return Err(PyValueError::new_err("invalid buffer payload size"))}
        let ptr=payload.as_ref().map_or(std::ptr::null(),|p|p.ptr());
        unsafe{(self.data)(target,size,ptr,usage)};self.check()
    }
    fn buffer_sub_data(&self,target:u32,offset:isize,payload:&Bound<'_,PyAny>)->PyResult<()> {
        let b=self.payload(payload)?;let size=b.len();
        if offset<0||offset.checked_add(size).is_none(){return Err(PyValueError::new_err("invalid buffer offset"))}
        unsafe{(self.sub)(target,offset,size,b.ptr())};self.check()
    }
    // Acquire/validate every exported buffer before mutating GL. The exporter
    // remains alive until all synchronous uploads finish; no GIL release.
    fn patch_rows(&self, vbo:u32, patches:Vec<(isize,Py<PyAny>)>, py:Python<'_>)->PyResult<usize>{
        if vbo==0{return Err(PyValueError::new_err("invalid buffer patches"))}
        let mut validated=Vec::with_capacity(patches.len());
        for (off,p) in &patches {let b=self.payload(p.bind(py))?;
            if *off<0||off.checked_add(b.len()).is_none(){return Err(PyValueError::new_err("invalid buffer patches"))}
            validated.push((*off,b));}
        unsafe{(self.bind_buffer)(0x8892,vbo)};
        let result=(||{self.check()?;for (off,b) in &validated {unsafe{(self.sub)(0x8892,*off,b.len(),b.ptr())};self.check()?;}Ok(validated.len())})();
        unsafe{(self.bind_buffer)(0x8892,0)};
        result.and_then(|n|{self.check()?;Ok(n)})
    }
    fn upload_static(&self,vao:u32,vbo:u32,payload:&Bound<'_,PyAny>)->PyResult<()> {
        if vao==0||vbo==0{return Err(PyValueError::new_err("invalid upload resource"))}
        let b=self.payload(payload)?;unsafe{(self.bind_vao)(vao);(self.bind_buffer)(0x8892,vbo)};
        let result=(||{self.check()?;unsafe{(self.data)(0x8892,b.len(),b.ptr(),0x88e4)};self.check()})();
        unsafe{(self.bind_buffer)(0x8892,0);(self.bind_vao)(0)};result.and_then(|_|self.check())
    }
    fn configure_depth(&self,id:u32,width:i32,height:i32,samples:i32)->PyResult<i32>{
        if id==0||width<=0||height<=0||samples<1{return Err(PyValueError::new_err("invalid depth storage"))}
        let mut actual=1;unsafe{
            (self.bind_rb)(0x8d41,id);
            if samples>1{(self.rb_ms)(0x8d41,samples,0x81a6,width,height);(self.rb_query)(0x8d41,0x8cab,&mut actual);}
            else{(self.rb_storage)(0x8d41,0x81a6,width,height);}
        }self.check()?;Ok(actual)
    }
    fn configure_color(&self,id:u32,width:i32,height:i32,samples:i32,format:u32,external:u32)->PyResult<()> {
        if id==0||width<=0||height<=0||samples<1||!matches!(format,0x8058|0x881a|0x822e){return Err(PyValueError::new_err("invalid color storage"))}
        unsafe{
            let target=if samples>1{0x9100}else{0x0de1};(self.bind_texture)(target,id);
            if samples>1{(self.tex_ms)(target,samples,format,width,height,1)}else{
                (self.tex_image)(target,0,format as i32,width,height,0,external,0x1406,std::ptr::null());
                (self.tex_param)(target,0x2801,0x2600);(self.tex_param)(target,0x2800,0x2600);(self.tex_param)(target,0x813d,0);
            }
        }self.check()
    }
    fn configure_framebuffer(&self,id:u32,texture:u32,depth:u32,samples:i32)->PyResult<u32>{
        if id==0||texture==0||depth==0||samples<1{return Err(PyValueError::new_err("invalid framebuffer names"))}
        let status=unsafe{
            (self.bind_fb)(0x8d40,id);
            (self.fb_tex)(0x8d40,0x8ce0,if samples>1{0x9100}else{0x0de1},texture,0);
            (self.fb_rb)(0x8d40,0x8d00,0x8d41,depth);
            (self.draw_buffer)(0x8ce0);(self.read_buffer)(0x8ce0);
            (self.fb_status)(0x8d40)
        };self.check()?;Ok(status)
    }

    /// Attach/detach the revealage texture to the accumulation framebuffer.
    fn attach_revealage(&self,id:u32,texture:u32,samples:i32)->PyResult<()> {
        if id==0||samples<1{return Err(PyValueError::new_err("invalid framebuffer name or sample count"))}
        unsafe{
            (self.bind_fb)(0x8d40,id);
            (self.fb_tex)(0x8d40,0x8ce1,if samples>1{0x9100}else{0x0de1},texture,0);
        }
        self.check()
    }

}
