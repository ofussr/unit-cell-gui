"""External controls for render-only updates; requires unit-cell-gui >= 0.2.5."""
from dataclasses import replace
import sys
import numpy as np
from unit_cell_gui import Atom, AtomComponent, Polyhedron, Scene, UnitCellViewer
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QCheckBox, QPushButton


def example_scene():
    c=AtomComponent('C','C','C',1,'#5588bb',.3)
    o=AtomComponent('O','O','O',1,'#dd8855',.3)
    vertices=np.array([(x,y,z) for x in (-.8,.8) for y in (-.8,.8) for z in (-.8,.8)])
    faces=((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
    poly=Polyhedron(np.zeros(3),vertices,faces,'cube','Cube',(c,))
    return Scene((Atom('C','C',(c,)),Atom('O','O',(o,))),[[0,0,0],[1,0,0]],[[0,1]],
                 [],[],np.eye(3),(poly,),base_radius=2)


class Host(QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle('unit-cell-gui: external partial updates')
        layout=QHBoxLayout(self); controls=QVBoxLayout(); layout.addLayout(controls)
        self.viewer=UnitCellViewer(); layout.addWidget(self.viewer,1)
        self.original=example_scene(); self.viewer.set_scene(self.original)
        self.viewer.set_display_options(show_polyhedra=True)
        self.x=QSlider(Qt.Orientation.Horizontal); self.x.setRange(-80,80)
        self.poly=QSlider(Qt.Orientation.Horizontal); self.poly.setRange(60,150); self.poly.setValue(100)
        self.opacity=QSlider(Qt.Orientation.Horizontal); self.opacity.setRange(0,100); self.opacity.setValue(64)
        self.visible=QCheckBox('Show atom 1'); self.visible.setChecked(True)
        self.bw=QCheckBox('Black and white'); reset=QPushButton('Reset partial changes')
        controls.addWidget(QLabel('Atom 1: displacement in X'))
        controls.addWidget(self.x); controls.addWidget(self.visible)
        controls.addWidget(QLabel('Polyhedron scale')); controls.addWidget(self.poly)
        controls.addWidget(QLabel('Polyhedron opacity (colour mode)')); controls.addWidget(self.opacity)
        controls.addWidget(self.bw); controls.addWidget(reset); controls.addStretch()
        self.x.valueChanged.connect(self.apply_batch)
        self.poly.valueChanged.connect(self.apply_batch)
        self.opacity.valueChanged.connect(lambda value:self.viewer.set_polyhedron_opacities({0:value/100}))
        self.visible.toggled.connect(self.apply_batch)
        self.bw.toggled.connect(lambda value:self.viewer.set_display_options(engraving=value))
        reset.clicked.connect(self.reset_changes)
        self.resize(1000,620)

    def apply_batch(self):
        center=self.original.atom_centers[1]+[self.x.value()/100,0,0]
        poly=replace(self.original.polyhedra[0],vertices=self.original.polyhedra[0].vertices*(self.poly.value()/100))
        self.viewer.update_scene(atom_positions={1:center},atom_visibility={1:self.visible.isChecked()},polyhedra={0:poly})

    def reset_changes(self):
        for control,value in ((self.x,0),(self.poly,100),(self.opacity,64),(self.visible,True)):
            control.blockSignals(True)
            if isinstance(control,QCheckBox): control.setChecked(value)
            else: control.setValue(value)
            control.blockSignals(False)
        self.viewer.reset_scene_updates()


if __name__=='__main__':
    app=QApplication(sys.argv); host=Host(); host.show(); sys.exit(app.exec())
