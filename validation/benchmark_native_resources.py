"""CPU submission including driver work, same current GL context, alternating backends."""
import argparse,json,os,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from OpenGL import GL as g
from unit_cell_gui import UnitCellViewer, opengl_viewer as renderer
from unit_cell_gui._native_resources import ResourceGL
from unit_cell_gui._weighted_oit import WeightedOIT,preserved_state

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=int,default=30);a=p.parse_args()
 app=QApplication([]);v=UnitCellViewer();v.resize(480,360);v.show();app.processEvents();v.grabFramebuffer();v.makeCurrent()
 try:
  resources={}
  for backend in ('python','rust'):
   with patch.dict(os.environ,{'UNIT_CELL_GUI_RESOURCE_BACKEND':backend}):resources[backend]=ResourceGL(v.context(),renderer)
  assert resources['rust'].native is not None,resources['rust'].error
  print(g.glGetString(g.GL_RENDERER),flush=True)
  b=int(g.glGenBuffers(1));vao=int(g.glGenVertexArrays(1));rng=np.random.default_rng(42)
  results=[]
  def legacy_static(r,data):
   r.glBindVertexArray(vao);r.glBindBuffer(g.GL_ARRAY_BUFFER,b);r.glBufferData(g.GL_ARRAY_BUFFER,data.nbytes,data,g.GL_STATIC_DRAW);r.glBindBuffer(g.GL_ARRAY_BUFFER,0);r.glBindVertexArray(0)
  def static(r,data):
   if r.native is None:legacy_static(r,data)
   else:r.upload_static(vao,b,data)
  def patch_rows(r,data,rows):
   # Actual widget path, including per-run validation and contiguous conversion.
   with patch.object(v,'_resource_gl',r):v._patch_rows(b,data,rows)
  def measure(name,operation,byte_count,run_count):
   timings={k:[] for k in resources}
   for _ in range(3):
    for r in resources.values():operation(r);g.glFinish()
   for i in range(a.samples):
    for backend in (('python','rust') if i%2==0 else ('rust','python')):
     g.glFinish();start=time.perf_counter_ns();operation(resources[backend]);elapsed=(time.perf_counter_ns()-start)/1e6;g.glFinish();timings[backend].append(elapsed)
   row={'case':name,'bytes':byte_count,'runs':run_count,'samples_ms':timings,'p50_ms':{k:float(np.median(t)) for k,t in timings.items()},'p95_ms':{k:float(np.percentile(t,95)) for k,t in timings.items()}}
   results.append(row);print(name,row['p50_ms'],flush=True)
  for n in (256,16384,262144):
   data=rng.random((n,16),dtype=np.float32)
   measure(f'upload_{n}',lambda r:static(r,data),data.nbytes,1)
  data=rng.random((16384,16),dtype=np.float32);static(resources['rust'],data)
  for n in (1,32,256):
   rows=list(range(0,n*8,8));measure(f'patch_{n}',lambda r:patch_rows(r,data,rows),n*64,n)
  def wboit(r):
   with preserved_state():
    oit=WeightedOIT(r);oit.ensure_size(480,360,4);oit.release_attachments()
  measure('wboit_allocate_release',wboit,480*360*4*20,0)
  g.glDeleteBuffers(1,[b]);g.glDeleteVertexArrays(1,[vao]);a.output.write_text(json.dumps({'metric':'CPU submission in software driver; glFinish outside measured interval','renderer':g.glGetString(g.GL_RENDERER).decode(),'version':g.glGetString(g.GL_VERSION).decode(),'samples':a.samples,'cases':results},indent=2)+'\n')
 finally:v.doneCurrent();v.close()
if __name__=='__main__':main()
