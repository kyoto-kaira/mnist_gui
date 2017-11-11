from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import model_creator


class LayerEditorBase(QWidget):
    def __init__(self, model_st, parent=None):
        super(LayerEditorBase, self).__init__(parent)
        self.model_st = model_st
        self.add_btn = QPushButton("追加", self)
        self.add_btn.clicked.connect(self.add_layer)

    def resizeEvent(self, event):
        self.add_btn.move(self.width() * 0.6, self.height() * 0.8)
        self.additional_resize(event)

    def additional_resize(self, event):
        pass

    def add_layer(self, event):
        pass


class Conv2dEditor(LayerEditorBase):
    def __init__(self, model_st, parent=None):
        super(Conv2dEditor, self).__init__(model_st, parent)

        # 文字の色を白くするパレットを用意
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.white)

        # フィルター数
        self.label_filters = QLabel("filters", self)
        self.label_filters.setPalette(palette)
        self.input_filters = QSpinBox(self)
        self.input_filters.setMinimum(1)
        self.input_filters.setMaximum(9999)
        self.input_filters.setValue(10)

        # 横のカーネルサイズ
        self.label_x = QLabel("x size", self)
        self.label_x.setPalette(palette)
        self.input_x = QSpinBox(self)
        self.input_x.setMinimum(1)
        self.input_x.setMaximum(28)
        self.input_x.setValue(3)

        # 縦のカーネルサイズ
        self.label_y = QLabel("y size", self)
        self.label_y.setPalette(palette)
        self.input_y = QSpinBox(self)
        self.input_y.setMinimum(1)
        self.input_y.setMaximum(28)
        self.input_y.setValue(3)

    def additional_resize(self, event):
        self.label_filters.move(self.width() * 0.1, self.height() * 0.1)
        self.input_filters.move(self.width() * 0.6, self.height() * 0.1)
        self.label_x.move(self.width() * 0.1, self.height() * 0.3)
        self.input_x.move(self.width() * 0.6, self.height() * 0.3)
        self.label_y.move(self.width() * 0.1, self.height() * 0.4)
        self.input_y.move(self.width() * 0.6, self.height() * 0.4)

    def add_layer(self):
        filters = self.input_filters.value()
        kernel_x = self.input_x.value()
        kernel_y = self.input_y.value()
        try:
            self.model_st.add_conv2d(filters, kernel_x, kernel_y)
            self.model_st.get_model()
        except RuntimeError as e:
            print(e)
