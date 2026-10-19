"""GL 3.3 prototype: shared depth, per-sample composition, with/without MSAA."""
import json
import numpy as np
from OpenGL import GL as g
from PySide6.QtWidgets import QApplication
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QImage
from unit_cell_gui.opengl_viewer import _create_program
from unit_cell_gui._weighted_oit import WeightedOIT, preserved_state, weighted_fragment

VERT = '''#version 330 core
uniform float z;
void main(){vec2 p[3]=vec2[3](vec2(-.79,-.73),vec2(.87,-.67),vec2(-.27,.81));gl_Position=vec4(p[gl_VertexID],z,1);}
'''
FRAG = '''#version 330 core
uniform vec4 colour;
out vec4 frag_colour;
void main(){frag_colour=colour;}
'''
class Prototype(QOpenGLWidget):
    def __init__(self,samples):
        super().__init__();fmt=self.format();fmt.setSamples(samples);self.setFormat(fmt)
        self.fast=False;self.layers=False;self.oit=WeightedOIT();self.resize(193,157)
    def initializeGL(self):
        self.normal=_create_program(VERT,FRAG)
        self.weighted=_create_program(VERT,weighted_fragment(FRAG))
        self.vao=int(g.glGenVertexArrays(1))
        self.oit.initialize()
    def draw(self,program,z,colour):
        g.glUseProgram(program);g.glBindVertexArray(self.vao)
        g.glUniform1f(g.glGetUniformLocation(program,'z'),z)
        g.glUniform4f(g.glGetUniformLocation(program,'colour'),*colour)
        g.glDrawArrays(g.GL_TRIANGLES,0,3)
    def paintGL(self):
        target=self.defaultFramebufferObject()
        g.glBindFramebuffer(g.GL_FRAMEBUFFER,target)
        self.samples=max(1,int(g.glGetIntegerv(g.GL_SAMPLES)))
        with preserved_state():
            if self.fast:
                vp=g.glGetIntegerv(g.GL_VIEWPORT)
                self.oit.ensure_size(int(vp[2]),int(vp[3]),self.samples)
                self.oit.begin_scene((1,1,1,1))
            else:
                g.glClearColor(1,1,1,1);g.glClearDepth(0)
                g.glDepthMask(True);g.glEnable(g.GL_DEPTH_TEST);g.glDepthFunc(g.GL_GREATER)
                g.glDisable(g.GL_BLEND);g.glEnable(g.GL_MULTISAMPLE)
                g.glClear(g.GL_COLOR_BUFFER_BIT|g.GL_DEPTH_BUFFER_BIT)
            self.draw(self.normal,.2,(1,0,0,1))
            if self.fast:
                for mode in (0,1):
                    (self.oit.begin_revealage if mode else self.oit.begin_accumulation)()
                    g.glUseProgram(self.weighted)
                    g.glUniform1i(g.glGetUniformLocation(self.weighted,'u_oit_pass'),mode)
                    g.glUniform1f(g.glGetUniformLocation(self.weighted,'u_weight_scale'),1)
                    if self.layers:
                        self.draw(self.weighted,-.4,(0,1,0,.5))  # hidden by opaque red
                        self.draw(self.weighted,.6,(0,0,1,.5))
                self.oit.composite(target)
        assert g.glGetError()==g.GL_NO_ERROR
    def pixels(self):
        image=self.grabFramebuffer().convertToFormat(QImage.Format.Format_RGBA8888)
        return np.frombuffer(image.bits(),np.uint8,count=image.sizeInBytes()).copy().reshape(image.height(),image.width(),4)
    def release(self):
        self.makeCurrent();self.oit.cleanup()
        g.glDeleteProgram(self.normal);g.glDeleteProgram(self.weighted);g.glDeleteVertexArrays(1,[self.vao]);self.doneCurrent()

if __name__=='__main__':
    app=QApplication([]);results=[]
    for samples in (0,4):
        w=Prototype(samples);w.show();app.processEvents()
        exact=w.pixels();w.fast=True;fast=w.pixels()
        np.testing.assert_array_equal(exact,fast)
        w.layers=True;layers=w.pixels()
        np.testing.assert_allclose(layers[layers.shape[0]//2,layers.shape[1]//2,:3],[128,0,128],atol=1)
        results.append({'requested_samples':samples,'actual_samples':w.samples,
                        'opaque_pixels_identical':True,'hidden_layer_rejected':True,
                        'attachment_bytes':w.oit.attachment_bytes,'size':w.oit.size})
        w.release();w.close()
    print(json.dumps(results,indent=2))
