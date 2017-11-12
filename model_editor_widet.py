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
        max_pool2d_editor = MaxPool2dEditor(self.model_creator, self)
        flatten_editor = FlattenEditor(self.model_creator, self)
        dense_editor = DenseEditor(self.model_creator, self)
        dropout_editor = DropOutEditor(self.model_creator, self)
        batch_normalization_editor = BatchNormalizationEditor(self.model_creator, self)
        activation_editor = ActivationEditor(self.model_creator, self)
        compile_editor = CompileEditor(self.model_creator, self)
        self.addTab(conv2d_editor, "畳み込み層")
        self.addTab(max_pool2d_editor, "プーリング")
        self.addTab(flatten_editor, "一次元化層")
        self.addTab(dense_editor, "全結合層")
        self.addTab(dropout_editor, "ドロップアウト")
        self.addTab(batch_normalization_editor, "norm")
        self.addTab(activation_editor, "活性化関数")
        self.addTab(compile_editor, "コンパイル")


class ModelDisplayWidget(QListView):
    """
    編集中のモデルを状態を出力するウィジェット
    """
    def __init__(self, model_creator, parent=None):
        super(ModelDisplayWidget, self).__init__(parent)
        self.list_model = QStringListModel()
        self.setModel(self.list_model)
        self.setFocusPolicy(Qt.NoFocus)

        self.model_creator = model_creator

    def update_notify(self):
        str_i_list = self.model_creator.get_str_i_list()
        str_list = [text for text, i in str_i_list]
        self.list_model.setStringList(str_list)
        print(str_i_list)


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
        self.reset_editor_model_btn = QPushButton("エディターのモデルを初期化", self)
        self.reset_editor_model_btn.clicked.connect(self.reset_editor_model)
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
        self.reset_editor_model_btn.move(self.width() * 0.1, self.height() * 0.9)
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

    def reset_editor_model(self):
        self.model_creator.clear()
