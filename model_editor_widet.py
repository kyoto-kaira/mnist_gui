from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from layer_editor_widgets import *
from model_creator import *

default_model_path = './model.hdf5'


class LayerEditorTab(QTabWidget):
    """
    レイヤーの追加を提供する、タブウィジェット
    """
    def __init__(self, model_creator, parent=None):
        super(LayerEditorTab, self).__init__(parent)
        self.model_creator = model_creator
        conv2d_editor = Conv2dEditor(self.model_creator, self)
        flatten_editor = FlattenEditor(self.model_creator, self)
        dense_editor = DenseEditor(self.model_creator, self)
        activation_editor = ActivationEditor(self.model_creator, self)
        compile_editor = CompileEditor(self.model_creator, self)
        self.addTab(conv2d_editor, "畳み込み層")
        self.addTab(flatten_editor, "一次元化層")
        self.addTab(dense_editor, "全結合層")
        self.addTab(activation_editor, "活性化関数")
        self.addTab(compile_editor, "コンパイル")


class ModelDisplayWidget(QTextBrowser):
    """
    編集中のモデルを状態を出力するウィジェット
    """
    def __init__(self, model_creator, parent=None):
        super(ModelDisplayWidget, self).__init__(parent)
        self.model_creator = model_creator

    def _append_shape(self, shape):
        self.append("  " + str(shape))

    def update_notify(self):
        self.clear()
        self.append("input")
        self._append_shape((28, 28, 1))
        for layer in self.model_creator:
            if isinstance(layer, ActivationLayer):
                self.append("activation ({})".format(layer.func_name))
            elif isinstance(layer, DenseLayer):
                self.append("Dense")
                self._append_shape(layer.output_shape)
            elif isinstance(layer, FlattenLayer):
                self.append("Flatten")
                self._append_shape(layer.output_shape)
            elif isinstance(layer, Conv2dLayer):
                self.append("Conv ({}, {}) x {}".format(layer.kernel[0],
                                                        layer.kernel[1],
                                                        layer.filters))
                self._append_shape(layer.output_shape)
            elif isinstance(layer, CompileLayer):
                self.append("output")
            else:
                self.append("unknown layer")


class ModelEditorWidget(QWidget):
    """
    モデルエディタータブの内容を表すウィジェット
    """
    def __init__(self, model, parent=None):
        super(ModelEditorWidget, self).__init__()
        self.model = model
        self.model_creator = ModelCreator()
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
        self.layer_editor = LayerEditorTab(self.model_creator, self)
        self.model_display = ModelDisplayWidget(self.model_creator, self)
        self.model_creator.set_changed_notify(self.model_display.update_notify)

    def resizeEvent(self, event):
        self.reset_model_btn.move(self.width() * 0.1, self.height() * 0.05)
        self.load_defo_btn.move(self.width() * 0.1, self.height() * 0.1)
        self.evaluate_btn.move(self.width() * 0.1, self.height() * 0.15)
        self.save_btn.move(self.width() * 0.1, self.height() * 0.2)
        self.layer_editor.move(self.width() * 0.1, self.height() * 0.3)
        self.layer_editor.resize(self.width() * 0.5, self.height() * 0.5)
        self.load_from_editor_btn.move(self.width() * 0.1, self.height() * 0.8)
        self.model_display.move(self.width() * 0.6, self.height() * 0.05)
        self.model_display.resize(self.width() * 0.5, self.height() * 0.9)

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
            model = self.model_creator.get_model()
            self.model.set_model(model)
        except RuntimeError as e:
            print(e)
