from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QImage, QPainter, QPen, qRgb, QBrush, QTextCursor
from PyQt5.QtWidgets import *
import numpy as np
import matplotlib.pyplot as plt

import threading
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.callbacks import LambdaCallback
from sklearn import datasets
from sklearn.model_selection import train_test_split


class MnistModel(threading.Thread):
    def __init__(self, logger, progress):
        super(MnistModel, self).__init__()
        self.logger = logger
        self.progress = progress

        self.mnist = datasets.fetch_mldata('MNIST original', data_home='.')
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self._set_train_and_test_data()

        self.model = None
        try:
            self.load('./model.hdf5')
        except:
            self.set_model()
        self.graph = tf.get_default_graph()

        self.learn_event = threading.Event()
        self.learn_event.clear()
        self._exit = False

    def _set_train_and_test_data(self):
        np.random.seed(0)
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

    def run(self):
        """学習を実行する。時間がかかるのでマルチスレッド化してある。"""
        while True:
            self.learn_event.wait()
            if self._exit:
                break
            epochs = 1
            batch_size = 1000
            num_batch = len(self.X_train) // batch_size
            if self.model is None:
                self.logger.append("no model")
                return

            self.progress.setValue(0)
            self.logger.append("start learning")

            def batch_end_out(epoch, logs):
                self.progress.setValue((epoch + 1) / num_batch * 100)
                self.logger.append(str("{}/{} {:.4f}".format(epoch + 1,
                                                             num_batch,
                                                             logs['acc'])))
                self.logger.moveCursor(QTextCursor.End)

            def epoch_end_out(epoch, logs):
                self.logger.append(str(logs))
                self.logger.moveCursor(QTextCursor.End)

            with self.graph.as_default():
                self.model.fit(self.X_train, self.Y_train,
                               validation_data=(self.X_test, self.Y_test),
                               epochs=epochs,
                               batch_size=batch_size,
                               callbacks=[LambdaCallback(on_batch_end=batch_end_out,
                                                         on_epoch_end=epoch_end_out)])

            if self._exit:
                break
            self.learn_event.clear()
            if self._exit:
                break

    def stop_learning(self):
        self.model.stop_training = True

    def kill(self):
        if self.model is None:
            return
        self.model.stop_training = True
        self._exit = True
        self.learn_event.set()

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


class HandWritingTab(QWidget):
    def __init__(self, model, parent=None):
        super(HandWritingTab, self).__init__()

        self.barGraph = BarGraph(self)
        self.scribbleArea = ScribbleArea(self.barGraph, model, parent=self)
        self.reset_btn = QPushButton("リセット", self)
        self.reset_btn.clicked.connect(self.reset_screen)

    def resizeEvent(self, event):
        self.reset_btn.move(self.width() * 0.01, self.height() * 0.01)

        self.scribbleArea.move(self.width() * 0.01, self.height() * 0.1)
        self.scribbleArea.resize(self.width() * 0.7, self.width() * 0.7)

        self.barGraph.move(self.width() * 0.75, self.height() * 0.1)
        self.barGraph.resize(self.width() * 0.2, self.height() * 0.8)

    def reset_screen(self, event):
        self.scribbleArea.clearImage()


class ModelEditorTab(QWidget):
    def __init__(self, parent=None):
        super(ModelEditorTab, self).__init__()


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setStyleSheet(appStyle)

        self.textArea = QTextBrowser(self)
        self.progress_bar = QProgressBar(self)

        self.model = MnistModel(self.textArea, self.progress_bar)

        self.HandWritingTab = HandWritingTab(self.model, self)
        self.ModelEditorTab = ModelEditorTab(self)
        self.tab = QTabWidget(self)
        self.tab.addTab(self.HandWritingTab, "tab1")
        self.tab.addTab(self.ModelEditorTab, "tab2")

        self.learn_btn = QPushButton("学習開始", self)
        self.learn_btn.clicked.connect(self.learn)
        self.stop_btn = QPushButton("学習中止", self)
        self.stop_btn.clicked.connect(self.stop_learning)

        self.createActions()
        self.createMenus()

        self.setWindowTitle("MNIST GUI")
        self.resize(700, 500)

        self.model.start()

    def resizeEvent(self, event):
        self.tab.move(self.width() * 0.25, self.height() * 0.05)
        self.tab.resize(self.width() * 0.7, self.height())

        self.learn_btn.move(self.width() * 0.01, self.height() * 0.3)
        self.stop_btn.move(self.width() * 0.01, self.height() * 0.35)

        self.progress_bar.move(self.width() * 0.01, self.height() * 0.9)
        self.progress_bar.resize(self.width() * 0.5, self.height() * 0.05)

        self.textArea.move(self.width() * 0.01, self.height() * 0.4)
        self.textArea.resize(self.width() * 0.18, self.height() * 0.5)

    def learn(self, event):
        self.model.learn_event.set()

    def stop_learning(self, event):
        self.model.stop_learning()

    def closeEvent(self, event):
        if self.exitWarn():
            self.model.kill()
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
                                      triggered=self.HandWritingTab.scribbleArea.clearImage)

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
            return True

        return False


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
