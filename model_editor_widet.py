from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from layer_editor_widgets import *
from model_creator import *
from one_line_info import global_one_line_info

default_model_path = './model.hdf5'


class LayerEditorWidget(QWidget):
    """
    レイヤーの追加を提供する、ウィジェット
    """
    def __init__(self, model_creator, parent=None):
        super(LayerEditorWidget, self).__init__(parent)
        self.model_creator = model_creator
        self.conv2d_editor = Conv2dEditor(self.model_creator)
        self.max_pool2d_editor = MaxPool2dEditor(self.model_creator)
        self.dense_editor = DenseEditor(self.model_creator)
        self.dropout_editor = DropOutEditor(self.model_creator)
        self.batch_normalization_editor = BatchNormalizationEditor(self.model_creator)
        self.activation_editor = ActivationEditor(self.model_creator)
        self.compile_editor = CompileEditor(self.model_creator)

        layout_0 = QVBoxLayout()
        layout_0.addWidget(self.conv2d_editor)
        layout_0.addWidget(self.max_pool2d_editor)
        layout_0.addWidget(self.dense_editor)
        layout_1 = QVBoxLayout()
        layout_1.addWidget(self.dropout_editor)
        layout_1.addWidget(self.batch_normalization_editor)
        layout_1.addWidget(self.activation_editor)
        layout_1.addWidget(self.compile_editor)
        layout = QHBoxLayout()
        layout.addLayout(layout_0)
        layout.addLayout(layout_1)
        self.setLayout(layout)


class ModelDisplayWidget(QListWidget):
    """
    編集中のモデルを状態を出力するウィジェット
    """
    def __init__(self, model_creator, parent=None):
        super(ModelDisplayWidget, self).__init__(parent)
        self.setFocusPolicy(Qt.NoFocus)

        self.model_creator = model_creator

    def update_notify(self):
        str_list = self.model_creator.get_str_list()
        self.clear()
        self.addItems(str_list)


class ModelEditorWidget(QWidget):
    """
    モデルエディタータブの内容を表すウィジェット
    """
    def __init__(self, model, parent=None):
        super(ModelEditorWidget, self).__init__()
        self.model = model
        self.model_creator = ModelCreator()
        self.load_defo_btn = QPushButton("学習済みのモデルをロード", self)
        self.load_defo_btn.clicked.connect(self.load_defo)
        self.evaluate_btn = QPushButton("モデルを評価", self)
        self.evaluate_btn.clicked.connect(self.evaluate_model)
        self.load_from_editor_btn = QPushButton("エディターからモデルをロード", self)
        self.load_from_editor_btn.clicked.connect(self.load_from_editor)
        self.reset_editor_model_btn = QPushButton("エディターのモデルを初期化", self)
        self.reset_editor_model_btn.clicked.connect(self.reset_editor_model)
        self.layer_editor = LayerEditorWidget(self.model_creator, self)
        self.model_display = ModelDisplayWidget(self.model_creator, self)
        self.model_creator.set_changed_notify(self.model_display.update_notify)
        self.reset_editor_model()

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0x2a, 0x2a, 0x2a))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.layer_editor.move(0, 0)
        self.layer_editor.setFixedSize(350, 500)

    def resizeEvent(self, event):
        self.load_defo_btn.move(self.width() * 0.6, self.height() * 0.1)
        self.evaluate_btn.move(self.width() * 0.6, self.height() * 0.15)

        self.reset_editor_model_btn.move(356, self.height() * 0.35)
        self.load_from_editor_btn.move(356, self.height() * 0.4)

        self.model_display.move(356, self.height() * 0.5)
        self.model_display.resize(200, self.height() * 0.45)

    def load_defo(self):
        try:
            self.model.load(default_model_path)
        except RuntimeError as e:
            global_one_line_info.send(str(e))

    def evaluate_model(self):
        global_one_line_info.send(str(self.model.report_evaluation()))

    def load_from_editor(self):
        try:
            model = self.model_creator.get_model()
            self.model.set_model(model)
        except RuntimeError as e:
            global_one_line_info.send(str(e))

    def reset_editor_model(self):
        self.model_creator.clear()
