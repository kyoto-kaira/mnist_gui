from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from model_creator import ActivationLayer


class LayerEditorBase(QGroupBox):
    def __init__(self, layer_name, model_creator, parent=None):
        super(LayerEditorBase, self).__init__(layer_name, parent)

        self.setStyleSheet("QGroupBox {border:1px solid gray; color :gray;}"
                           "QLabel {color: white;}")

        self.model_creator = model_creator


class DenseEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(DenseEditor, self).__init__("Dense", model_creator, parent)

        # ユニット数
        label_units = QLabel("units")
        self.input_units = QSpinBox()
        self.input_units.setMinimum(1)
        self.input_units.setMaximum(9999)
        self.input_units.setValue(100)

        # ボタン
        add_btn = QPushButton("追加")
        add_btn.clicked.connect(self.add_layer)

        layout_units = QHBoxLayout()
        layout_units.addWidget(label_units)
        layout_units.addWidget(self.input_units)
        layout = QVBoxLayout()
        layout.addLayout(layout_units)
        layout.addWidget(add_btn)
        self.setLayout(layout)

    def add_layer(self, event):
        units = self.input_units.value()
        try:
            self.model_creator.add_dense(units)
        except RuntimeError as e:
            print(e)


class ActivationEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(ActivationEditor, self).__init__("Activation", model_creator, parent)

        self.combo = QComboBox()
        for func_name in ActivationLayer.get_func_set():
            self.combo.addItem(func_name)

        # ボタン
        add_btn = QPushButton("追加")
        add_btn.clicked.connect(self.add_layer)

        layout = QVBoxLayout()
        layout.addWidget(self.combo)
        layout.addWidget(add_btn)
        self.setLayout(layout)

    def add_layer(self, event):
        try:
            self.model_creator.add_activation(self.combo.currentText())
        except RuntimeError as e:
            print(e)


class DropOutEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(DropOutEditor, self).__init__("Dropout", model_creator, parent)

        # ドロップアウト率
        label_ratio = QLabel("dropout ratio")
        self.input_ratio = QLineEdit()
        self.input_ratio.setValidator(QDoubleValidator(0.0, 1.0, 5))
        self.input_ratio.setText("0.5")

        # ボタン
        add_btn = QPushButton("追加")
        add_btn.clicked.connect(self.add_layer)

        layout = QVBoxLayout()
        layout.addWidget(label_ratio)
        layout.addWidget(self.input_ratio)
        layout.addWidget(add_btn)
        self.setLayout(layout)

    def add_layer(self, event):
        try:
            self.model_creator.add_dropout(self.input_ratio.text())
        except RuntimeError as e:
            print(e)


class Conv2dEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(Conv2dEditor, self).__init__("Conv", model_creator, parent)

        # フィルター数
        label_filters = QLabel("filters", self)
        self.input_filters = QSpinBox(self)
        self.input_filters.setMinimum(1)
        self.input_filters.setMaximum(9999)
        self.input_filters.setValue(10)

        # 横のカーネルサイズ
        label_x = QLabel("x size")
        self.input_x = QSpinBox(self)
        self.input_x.setMinimum(1)
        self.input_x.setMaximum(28)
        self.input_x.setValue(3)

        # 縦のカーネルサイズ
        label_y = QLabel("y size")
        self.input_y = QSpinBox()
        self.input_y.setMinimum(1)
        self.input_y.setMaximum(28)
        self.input_y.setValue(3)

        # ボタン
        add_btn = QPushButton("追加")
        add_btn.clicked.connect(self.add_layer)

        layout_filter = QHBoxLayout()
        layout_filter.addWidget(label_filters)
        layout_filter.addWidget(self.input_filters)
        layout_x = QHBoxLayout()
        layout_x.addWidget(label_x)
        layout_x.addWidget(self.input_x)
        layout_y = QHBoxLayout()
        layout_y.addWidget(label_y)
        layout_y.addWidget(self.input_y)
        layout = QVBoxLayout()
        layout.addLayout(layout_filter)
        layout.addLayout(layout_x)
        layout.addLayout(layout_y)
        layout.addWidget(add_btn)
        self.setLayout(layout)

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
        super(MaxPool2dEditor, self).__init__("MaxPool", model_creator, parent)

        # 横のプールサイズ
        label_x = QLabel("x size")
        self.input_x = QSpinBox()
        self.input_x.setMinimum(1)
        self.input_x.setMaximum(28)
        self.input_x.setValue(3)

        # 縦のプールサイズ
        label_y = QLabel("y size")
        self.input_y = QSpinBox()
        self.input_y.setMinimum(1)
        self.input_y.setMaximum(28)
        self.input_y.setValue(3)

        # ボタン
        add_btn = QPushButton("追加")
        add_btn.clicked.connect(self.add_layer)

        layout_x = QHBoxLayout()
        layout_x.addWidget(label_x)
        layout_x.addWidget(self.input_x)
        layout_y = QHBoxLayout()
        layout_y.addWidget(label_y)
        layout_y.addWidget(self.input_y)
        layout = QVBoxLayout()
        layout.addLayout(layout_x)
        layout.addLayout(layout_y)
        layout.addWidget(add_btn)
        self.setLayout(layout)

    def add_layer(self):
        pool_x = self.input_x.value()
        pool_y = self.input_y.value()
        try:
            self.model_creator.add_max_pool2d(pool_x, pool_y)
        except RuntimeError as e:
            print(e)


class BatchNormalizationEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(BatchNormalizationEditor, self).__init__("Batch Normalization",
                                                       model_creator,
                                                       parent)

        # ボタン
        add_btn = QPushButton("追加")
        add_btn.clicked.connect(self.add_layer)

        layout = QHBoxLayout()
        layout.addWidget(add_btn)
        self.setLayout(layout)

    def add_layer(self, event):
        try:
            self.model_creator.add_batch_normalization()
        except RuntimeError as e:
            print(e)


class CompileEditor(LayerEditorBase):
    def __init__(self, model_creator, parent=None):
        super(CompileEditor, self).__init__("Compile",
                                            model_creator,
                                            parent)

        # 層を削除するボタン
        delete_layer_btn = QPushButton("最後の層を削除")
        delete_layer_btn.clicked.connect(self.delete_layer)

        # ボタン
        compile_btn = QPushButton("モデルをコンパイル")
        compile_btn.clicked.connect(self.add_layer)

        layout = QVBoxLayout()
        layout.addWidget(delete_layer_btn)
        layout.addWidget(compile_btn)
        self.setLayout(layout)

    def delete_layer(self, event):
        try:
            self.model_creator.delete_last_layer()
        except RuntimeError as e:
            print(e)

    def add_layer(self, event):
        try:
            self.model_creator.add_compile()
        except RuntimeError as e:
            print(e)
