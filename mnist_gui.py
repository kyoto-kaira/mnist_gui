from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QImage, QPainter, QPen, qRgb
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

try:
    if True:
        model = load_model('./model.hdf5')
        model.summary()
    else:
        raise 'hoge'
except:
    '''
    データの生成
    '''
    mnist = datasets.fetch_mldata('MNIST original', data_home='.')

    n = len(mnist.data)
    N = n #10000  # MNISTの一部を使う
    indices = np.random.permutation(range(n))[:N]  # ランダムにN枚を選択

    X = mnist.data[indices]
    X = X.reshape(-1, 28, 28, 1)
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

    if False:
        toshow = X_test.reshape(-1, 28, 28)
        for i in range(100):
            for j in range(10):
                print("{}: {:.4f}".format(j, Y_test[i][j]))
            plt.imshow(toshow[i])
            plt.show()

    '''
    モデル設定
    '''
    n_in = len(X[0])  # 784
    n_hidden = 200
    # n_hidden = 4000
    n_out = len(Y[0])  # 10

    model = Sequential()

    model.add(Convolution2D(15,
                            (3, 3),
                            input_shape=(28, 28, 1),
                            activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(15,
                            (3, 3),
                            activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())

    model.add(BatchNormalization())

    model.add(Dense(n_hidden))



    model.add(Dropout(0.5))

    model.add(Dense(n_out))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    model.summary()

    '''
    モデル学習
    '''
    epochs = 10
    batch_size = 200

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    '''
    予測精度の評価
    '''
    loss_and_metrics = model.evaluate(X_test, Y_test)
    print(loss_and_metrics)

    model.save('./model.hdf5')


class ScribbleArea(QWidget):
    def __init__(self, text_output, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 40
        self.myPenColor = Qt.black
        self.image = QImage()
        self.lastPoint = QPoint()
        self.textOutput = text_output

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
        y_ = model.predict(imageArray)
        y_ = y_.reshape(10)
        self.textOutput.clear()
        for i in range(10):
            print("{}: {:.4f}".format(i, y_[i]))
            self.textOutput.append("{}: {:.4f}\n".format(i, y_[i]))


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

appStyle="""
QMainWindow{
background-color: #333333;
}
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setStyleSheet(appStyle)

        layout = QHBoxLayout()
        self.text = QTextBrowser(self)
        self.text.append("hogehoge")
        self.scribbleArea = ScribbleArea(self.text)
        self.setCentralWidget(self.scribbleArea)
        self.reset_btn = QPushButton("リセット", self)
        self.reset_btn.clicked.connect(self.reset_screen)
        layout.addWidget(self.scribbleArea)
        layout.addWidget(self.reset_btn)
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

        self.reset_btn.move(self.width() * 0.01, self.height() * 0.5)

        self.text.move(self.width() * 0.75, self.height() * 0.1)
        self.text.resize(self.width() * 0.2, self.height() * 0.8)

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
