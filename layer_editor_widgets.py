from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from model_creator import ActivationLayer


class LayerEditorBase(QWidget):
    def __init__(self, model_creator, parent=None):
        super(LayerEditorBase, self).__init__(parent)

        # 文字の色を白くするパレットを用意
        self.palette = QPalette()
        self.palette.setColor(QPalette.WindowText, Qt.white)

        self.model_creator = model_creator
        self.add_btn = QPushButton("追加", self)
        self.add_btn.clicked.connect(self.add_layer)

    def resizeEvent(self, event):
        self.add_btn.move(self.width() * 0.6, self.height() * 0.8)
        self.additional_resize(event)

    def additional_resize(self, event):
        pass

    def add_layer(self, event):
        pass


class DenseEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(DenseEditor, self).__init__(model_creator, parent)

        # ユニット数
        self.label_units = QLabel("units", self)
        self.label_units.setPalette(self.palette)
        self.input_units = QSpinBox(self)
        self.input_units.setMinimum(1)
        self.input_units.setMaximum(9999)
        self.input_units.setValue(100)

    def additional_resize(self, event):
        self.label_units.move(self.width() * 0.1, self.height() * 0.3)
        self.input_units.move(self.width() * 0.6, self.height() * 0.3)

    def add_layer(self, event):
        units = self.input_units.value()
        try:
            self.model_creator.add_dense(units)
        except RuntimeError as e:
            print(e)


class ActivationEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(ActivationEditor, self).__init__(model_creator, parent)
        self.combo = QComboBox(self)
        for func_name in ActivationLayer.get_func_set():
            self.combo.addItem(func_name)

    def additional_resize(self, event):
        self.combo.move(self.width() * 0.2, self.height() * 0.2)

    def add_layer(self, event):
        try:
            self.model_creator.add_activation(self.combo.currentText())
        except RuntimeError as e:
            print(e)


class DropOutEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(DropOutEditor, self).__init__(model_creator, parent)

        # ドロップアウト率
        self.label_ratio = QLabel("dropout ratio", self)
        self.label_ratio.setPalette(self.palette)
        self.input_ratio = QLineEdit(self)
        self.input_ratio.setValidator(QDoubleValidator(0.0, 1.0, 10))
        self.input_ratio.setText("0.5")

    def additional_resize(self, event):
        self.label_ratio.move(self.width() * 0.1, self.height() * 0.3)
        self.input_ratio.move(self.width() * 0.6, self.height() * 0.3)

    def add_layer(self, event):
        try:
            self.model_creator.add_dropout(self.input_ratio.text())
        except RuntimeError as e:
            print(e)



class Conv2dEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(Conv2dEditor, self).__init__(model_creator, parent)

        # フィルター数
        self.label_filters = QLabel("filters", self)
        self.label_filters.setPalette(self.palette)
        self.input_filters = QSpinBox(self)
        self.input_filters.setMinimum(1)
        self.input_filters.setMaximum(9999)
        self.input_filters.setValue(10)

        # 横のカーネルサイズ
        self.label_x = QLabel("x size", self)
        self.label_x.setPalette(self.palette)
        self.input_x = QSpinBox(self)
        self.input_x.setMinimum(1)
        self.input_x.setMaximum(28)
        self.input_x.setValue(3)

        # 縦のカーネルサイズ
        self.label_y = QLabel("y size", self)
        self.label_y.setPalette(self.palette)
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
            self.model_creator.add_conv2d(filters, kernel_x, kernel_y)
        except RuntimeError as e:
            print(e)


class MaxPool2dEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(MaxPool2dEditor, self).__init__(model_creator, parent)

        # 横のプールサイズ
        self.label_x = QLabel("x size", self)
        self.label_x.setPalette(self.palette)
        self.input_x = QSpinBox(self)
        self.input_x.setMinimum(1)
        self.input_x.setMaximum(28)
        self.input_x.setValue(3)

        # 縦のプールサイズ
        self.label_y = QLabel("y size", self)
        self.label_y.setPalette(self.palette)
        self.input_y = QSpinBox(self)
        self.input_y.setMinimum(1)
        self.input_y.setMaximum(28)
        self.input_y.setValue(3)

    def additional_resize(self, event):
        self.label_x.move(self.width() * 0.1, self.height() * 0.3)
        self.input_x.move(self.width() * 0.6, self.height() * 0.3)
        self.label_y.move(self.width() * 0.1, self.height() * 0.4)
        self.input_y.move(self.width() * 0.6, self.height() * 0.4)

    def add_layer(self):
        pool_x = self.input_x.value()
        pool_y = self.input_y.value()
        try:
            self.model_creator.add_max_pool2d(pool_x, pool_y)
        except RuntimeError as e:
            print(e)


class BatchNormalizationEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(BatchNormalizationEditor, self).__init__(model_creator, parent)

    def add_layer(self, event):
        try:
            self.model_creator.add_batch_normalization()
        except RuntimeError as e:
            print(e)


class CompileEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(CompileEditor, self).__init__(model_creator, parent)

    def add_layer(self, event):
        try:
            self.model_creator.add_compile()
        except RuntimeError as e:
            print(e)
