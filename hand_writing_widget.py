from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np
import matplotlib.pyplot as plt




class BarGraph(QWidget):
    def __init__(self, parent=None):
        super(BarGraph, self).__init__(parent)
        self.values = np.zeros(10)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBackgroundMode(Qt.OpaqueMode)
        painter.setBackground(Qt.white)
        dirty_rect = event.rect()
        painter.setBrush(Qt.white)
        painter.drawRect(dirty_rect)
        painter.setPen(Qt.black)
        painter.setBrush(QBrush(Qt.green))
        y_step = 20
        for i, v in enumerate(self.values):
            painter.drawRect(15, y_step * i + 20, 100 * v, 5)
            painter.drawText(5, y_step * i + 25, "{}".format(i))
            painter.drawText(120, y_step * i + 25, "{:.4f}".format(v))

    def setValues(self, values):
        self.values = values
        self.update()


class ScribbleArea(QWidget):
    def __init__(self, bar_output, model, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 40
        self.myPenColor = Qt.black
        self.image = QImage()
        self.lastPoint = QPoint()
        self.barOutput = bar_output
        self.model = model

    def getProcessedImage(self):
        self.resizeImage(self.image, self.size())
        scaled_image = self.image.smoothScaled(28, 28)
        height = scaled_image.height()
        width = scaled_image.width()
        black_set_x = []
        black_set_y = []
        image_array = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                image_array[y][x] = 255 - scaled_image.pixelColor(x, y).lightness()
                if image_array[y][x] >= 254:
                    black_set_y.append(y)
                    black_set_x.append(x)
        if (len(black_set_x) is not 0) or (len(black_set_y) is not 0):
            mid_x = (max(black_set_x) + min(black_set_x)) // 2
            mid_y = (max(black_set_y) + min(black_set_y)) // 2
        else:
            mid_x = 0
            mid_y = 0

        translated_image = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                y_ = y - mid_y + height // 2
                x_ = x - mid_x + width // 2
                if 0 <= x_ < 28 and 0 <= y_ < 28:
                    translated_image[y_][x_] = image_array[y][x]
        return translated_image.reshape((1, 28, 28, 1))

    def outputAcc(self):
        image_array = self.getProcessedImage()
        y = self.model.predict(image_array)
        self.barOutput.setValues(y)
        # for i in range(10):
        #     print("{}: {:.4f}".format(i, y[i]))

    def showImage(self):
        image_array = self.getProcessedImage().reshape(28, 28)
        plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
        plt.pause(0.01)

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    def paintEvent(self, event):
        painter = QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)
        self.outputAcc()

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 128, self.image.width())
            newHeight = max(self.height() + 128, self.image.height())
            self.resizeImage(self.image, QSize(newWidth, newHeight))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        painter.setPen(QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine,
                            Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = self.myPenWidth / 2 + 2
        self.update(QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QImage(newSize, QImage.Format_RGB32)
        newImage.fill(qRgb(255, 255, 255))
        painter = QPainter(newImage)
        painter.drawImage(QPoint(0, 0), image)
        self.image = newImage

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth


class HandWritingWidget(QWidget):
    def __init__(self, model, parent=None):
        super(HandWritingWidget, self).__init__()

        self.barGraph = BarGraph(self)
        self.scribbleArea = ScribbleArea(self.barGraph, model, parent=self)
        self.reset_btn = QPushButton("画面をクリア (Space)", self)
        self.reset_btn.clicked.connect(self.reset_screen)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0x2a, 0x2a, 0x2a))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        self.lbl = QLabel(self)

    def resizeEvent(self, event):
        self.initUI(self.width(), self.height())
        self.lbl.setVisible = False
        self.initUI(self.width(), self.height())
        self.reset_btn.move(self.width() * 0.05, self.height() * 0.025)

        self.scribbleArea.move(self.width() * 0.05, self.height() * 0.1)
        self.scribbleArea.resize(self.width() * 0.575, self.width() * 0.575)

        self.barGraph.move(self.width() * 0.65, self.height() * 0.1)
        self.barGraph.resize(200, 220)

        # 描画スペースのサイズに合わせて、ペンのサイズを自動設定
        self.scribbleArea.setPenWidth(self.width() * 0.08)

    def initUI(self, width, height):
        # QPixmapオブジェクト作成
        pixmap = QPixmap("./logo.png")

        # ラベルを作ってその中に画像を置く
        self.lbl.setPixmap(pixmap)
        self.lbl.move(width - 400, height - 133)
        self.lbl.show()


    def reset_screen(self, event):
        self.scribbleArea.clearImage()
