from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QImage, QPainter, QPen, qRgb, QBrush
from PyQt5.QtWidgets import *
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model
from sklearn import datasets
from sklearn.model_selection import train_test_split


class MnistModel:
    def __init__(self, logger):
        self.logger = logger
        self.mnist = datasets.fetch_mldata('MNIST original', data_home='.')
        self.shuffle()
        self.model = None

    def shuffle(self):
        n = len(self.mnist.data)
        indices = np.random.permutation(range(n))[:n]

        x = self.mnist.data[indices]
        x = x.reshape(-1, 28, 28, 1)
        y = self.mnist.target[indices]
        y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

        self.X_train, self.X_test, self.Y_train, self.Y_test\
            = train_test_split(x, y, train_size=0.8)

    def load(self, s):
        self.model = load_model(s)

    def learn(self, epochs, batch_size):
        if self.model is None:
            self.logger.append("no model")
            return
        self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, image):
        if self.model is None:
            return
        return self.model.predict(image).reshape(10)

    def set_model(self):
        self.model = Sequential()

        self.model.add(Convolution2D(15,
                                     (3, 3),
                                     input_shape=(28, 28, 1),
                                     activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Convolution2D(15,
                                     (3, 3),
                                     activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Flatten())

        self.model.add(BatchNormalization())

        self.model.add(Dense(200))

        self.model.add(Dropout(0.5))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.01),
                           metrics=['accuracy'])
        self.model.summary()


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
        painter.setPen(Qt.red)
        painter.setBrush(QBrush(Qt.darkBlue))
        for i, v in enumerate(self.values):
            painter.drawRect(15, 40 * i + 20, 100 * v, 5)
            painter.drawText(5, 40 * i + 25, "{}".format(i))
            painter.drawText(120, 40 * i + 25, "{:.4f}".format(v))

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

    def showacc(self):
        self.resizeImage(self.image, self.size())
        scaledImage = self.image.smoothScaled(28, 28)
        height = scaledImage.height()
        width = scaledImage.width()
        imageArray = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                imageArray[y][x] = 255 - scaledImage.pixelColor(x, y).lightness()
        #print(self.image.pixelColor(0, 0).lightness())
        #print(256 * 256 * 256 * 256 - 1)
        #plt.imshow(imageArray)
        #plt.show()
        #plt.pause(.01)
        imageArray = imageArray.reshape((1, 28, 28, 1))
        y_ = self.model.predict(imageArray)
        self.barOutput.setValues(y_)
        for i in range(10):
            print("{}: {:.4f}".format(i, y_[i]))

    def showImage(self):
        self.showacc()

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
        self.showacc()

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


appStyle = """
QMainWindow{
background-color: #333333;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setStyleSheet(appStyle)

        layout = QHBoxLayout()
        self.barGraph = BarGraph(self)
        self.textArea = QTextBrowser(self)

        self.model = MnistModel(self.textArea)
        self.model.load('./model.hdf5')

        self.scribbleArea = ScribbleArea(self.barGraph, self.model)
        self.setCentralWidget(self.scribbleArea)
        self.reset_btn = QPushButton("リセット", self)
        self.reset_btn.clicked.connect(self.reset_screen)
        layout.addWidget(self.reset_btn)
        layout.addWidget(self.textArea)
        layout.addWidget(self.scribbleArea)
        layout.addWidget(self.barGraph)
        self.setLayout(layout)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("MNIST GUI")
        self.resize(700, 500)


    def reset_screen(self, event):
        self.scribbleArea.clearImage()

    def resizeEvent(self, event):
        self.scribbleArea.move(self.width() * 0.2, self.height() * 0.1)
        self.scribbleArea.resize(self.height() * 0.7, self.height() * 0.7)

        self.reset_btn.move(self.width() * 0.01, self.height() * 0.2)

        self.textArea.move(self.width() * 0.01, self.height() * 0.4)
        self.textArea.resize(self.width() * 0.18, self.height() * 0.5)

        self.barGraph.move(self.width() * 0.75, self.height() * 0.1)
        self.barGraph.resize(self.width() * 0.2, self.height() * 0.8)

    def closeEvent(self, event):
        if self.exitWarn():
            event.accept()
        else:
            event.ignore()

    def showImage(self):
        self.scribbleArea.showImage()

    def penColor(self):
        newColor = QColorDialog.getColor(self.scribbleArea.penColor())
        if newColor.isValid():
            self.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QInputDialog.getInt(self, "MNIST GUI",
                                           "Select pen width:", self.scribbleArea.penWidth(), 1, 50, 1)
        if ok:
            self.scribbleArea.setPenWidth(newWidth)

    def about(self):
        QMessageBox.about(self, "About MNIST GUI",
                          "<p>The <b>MNIST GUI</b> provides hand-drawing tests.")

    def createActions(self):
        self.showAct = QAction("&showImage", self,
                               triggered=self.showImage)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               triggered=self.close)

        self.penColorAct = QAction("&Pen Color...", self,
                                   triggered=self.penColor)

        self.penWidthAct = QAction("Pen &Width...", self,
                                   triggered=self.penWidth)

        self.clearScreenAct = QAction("&Clear Screen", self, shortcut="Ctrl+L",
                                      triggered=self.scribbleArea.clearImage)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                                  triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.showAct)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAct)

        optionMenu = QMenu("&Options", self)
        optionMenu.addAction(self.penColorAct)
        optionMenu.addAction(self.penWidthAct)
        optionMenu.addSeparator()
        optionMenu.addAction(self.clearScreenAct)

        helpMenu = QMenu("&Help", self)
        helpMenu.addAction(self.aboutAct)
        helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(fileMenu)
        self.menuBar().addMenu(optionMenu)
        self.menuBar().addMenu(helpMenu)

    def exitWarn(self):
        ret = QMessageBox.warning(self, "MNIST GUID",
                                  "Are you sure to exit?",
                                  QMessageBox.Ok | QMessageBox.Cancel)
        if ret == QMessageBox.Ok:
            return True
        else:
            return False

    def saveFile(self, fileFormat):

        print("fileFormat:%s" % fileFormat)
        initialPath = QDir.currentPath() + '/untitled.' + fileFormat

        fileName, _ = QFileDialog.getSaveFileName(self, "Save As", initialPath,
                                                  "%s Files (*.%s);;All Files (*)" % (fileFormat.upper(), fileFormat))

        if fileName:
            print("dummy saveFile")
            return True;

        return False


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
