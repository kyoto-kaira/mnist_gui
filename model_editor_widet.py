from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from layer_editor_widgets import *
import model_creator

default_model_path = './model.hdf5'


class LayerEditorTab(QTabWidget):
    def __init__(self, parent=None):
        super(LayerEditorTab, self).__init__(parent)
        self.model_st = model_creator.ModelCreator()
        conv2d_editor = Conv2dEditor(self.model_st, self)
        flatten_editor = FlattenEditor(self.model_st, self)
        dense_editor = DenseEditor(self.model_st, self)
        activation_editor = ActivationEditor(self.model_st, self)
        compile_editor = CompileEditor(self.model_st, self)
        self.addTab(conv2d_editor, "畳み込み層")
        self.addTab(flatten_editor, "一次元化層")
        self.addTab(dense_editor, "全結合層")
        self.addTab(activation_editor, "活性化関数")
        self.addTab(compile_editor, "コンパイル")


class ModelEditorWidget(QWidget):
    def __init__(self, model, parent=None):
        super(ModelEditorWidget, self).__init__()
        self.model = model
        self.reset_model_btn = QPushButton("モデルを初期化", self)
        self.reset_model_btn.clicked.connect(self.reset_model)
        self.load_defo_btn = QPushButton("学習済みのモデルをロード", self)
        self.load_defo_btn.clicked.connect(self.load_defo)
        self.evaluate_btn = QPushButton("モデルを評価", self)
        self.evaluate_btn.clicked.connect(self.evaluate_model)
        self.save_btn = QPushButton("モデルを保存", self)
        self.save_btn.clicked.connect(self.save_model)
        self.load_from_editor_btn = QPushButton("エディターからモデルをロード", self)
        self.load_from_editor_btn.clicked.connect(self.load_from_editor)
        self.layer_editor = LayerEditorTab(self)

    def resizeEvent(self, event):
        self.reset_model_btn.move(self.width() * 0.1, self.height() * 0.05)
        self.load_defo_btn.move(self.width() * 0.1, self.height() * 0.1)
        self.evaluate_btn.move(self.width() * 0.1, self.height() * 0.15)
        self.save_btn.move(self.width() * 0.1, self.height() * 0.2)
        self.layer_editor.move(self.width() * 0.1, self.height() * 0.3)
        self.layer_editor.resize(self.width() * 0.5, self.height() * 0.5)
        self.load_from_editor_btn.move(self.width() * 0.1, self.height() * 0.8)

    def reset_model(self):
        try:
            self.model.set_model()
        except RuntimeError as e:
            print(e)

    def load_defo(self):
        try:
            self.model.load(default_model_path)
        except RuntimeError as e:
            print(e)

    def evaluate_model(self):
        print(self.model.report_evaluation())

    def save_model(self):
        try:
            self.model.save(default_model_path)
        except RuntimeError as e:
            print(e)

    def load_from_editor(self):
        try:
            model = self.layer_editor.model_st.get_model()
            self.model.set_model(model)
        except RuntimeError as e:
            print(e)
