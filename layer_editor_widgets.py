from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Conv2dEditor(QWidget):
    def __init__(self, parent=None):
        super(Conv2dEditor, self).__init__(parent)

        # 文字の色を白くするパレットを用意
        palette = QPalette()
        palette.setColor(QPalette.WindowText, Qt.white)

        # 横のカーネルサイズ
        self.label_x = QLabel("x size", self)
        self.label_x.setPalette(palette)
        self.input_x = QSpinBox(self)
        self.input_x.setMinimum(1)
        self.input_x.setMaximum(28)

        # 縦のカーネルサイズ
        self.label_y = QLabel("y size", self)
        self.label_y.setPalette(palette)
        self.input_y = QSpinBox(self)
        self.input_y.setMinimum(1)
        self.input_y.setMaximum(28)

    def resizeEvent(self, event):
        self.label_x.move(self.width() * 0.1, self.height() * 0.2)
        self.input_x.move(self.width() * 0.6, self.height() * 0.2)
        self.label_y.move(self.width() * 0.1, self.height() * 0.4)
        self.input_y.move(self.width() * 0.6, self.height() * 0.4)
