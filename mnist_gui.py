from PyQt5.QtCore import QDir, Qt
from PyQt5.QtGui import QPalette, QPixmap
from PyQt5.QtWidgets import *

from mnist_model import MnistModel
from hand_writing_widget import HandWritingWidget
from model_editor_widet import ModelEditorWidget
from ranking_widget import RankingWidget
from one_line_info import global_one_line_info


appStyle = """
QMainWindow {
background-color: #333333;
}
QTabWidget::pane {
color: #aaaaaa;
background-color: #2a2a2a;
}
QTabBar::tab:selected {
border-color: #2a2a2a;
border-style: solid;
border-width: 1px 1px 0px 1px;
padding:10px;
color: #aaaaaa;
background-color: #2a2a2a;
}
QTabBar::tab:!selected {
border-color: grey;
border-style: solid;
border-width: 0px 1px 0px 0px;
padding:10px;
color: #aaaaaa;
background-color: #666666;
}
QStatusBar {
color: #aaaaaa;
background-color: #333333;
}
QPushButton{
width:150px;
height:30px;
font-size:12px;
text-decoration:none;
text-align:center;
padding: 1px 5 1px;
color:#fff;
background-color:#49a9d4;
border-radius:12px;
}
QPushButton:hover{
width:150px;
height:30px;
font-size:12px;
text-decoration:none;
text-align:center;
padding: 1px 5 1px;
color:#fff;
background-color:#0075A9;
border-radius:12px;
}
#rankingWidget {
background-color: #333333;
}

"""


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setStyleSheet(appStyle)

        self.textArea = QTextBrowser(self)
        self.progress_bar = QProgressBar(self)

        self.model = MnistModel(self.textArea, self.progress_bar)

        self.HandWriting = HandWritingWidget(self.model, self)
        self.ModelEditor = ModelEditorWidget(self.model, self)
        self.Ranking = RankingWidget(self.model, self)
        self.tab = QTabWidget(self)
        self.tab.addTab(self.HandWriting, "Hand Writing")
        self.tab.addTab(self.ModelEditor, "Model Editor")
        self.tab.addTab(self.Ranking, "Ranking")

        self.model.set_update_bar_func(self.HandWriting.scribbleArea.outputAcc)

        self.learn_btn = QPushButton("学習開始", self)
        self.learn_btn.clicked.connect(self.learn)
        self.stop_btn = QPushButton("学習中止", self)
        self.stop_btn.clicked.connect(self.stop_learning)

        self.createActions()
        self.createMenus()
        global_one_line_info.set_destination(self.statusBar().showMessage)

        self.setWindowTitle("MNIST GUI")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        self.model.start()
        #self.initUI(self.width, self.height)



    def resizeEvent(self, event):
        self.tab.move(self.width() * 0.2, self.height() * 0.05)
        self.tab.resize(self.width() * 0.75, self.height() * 0.9)

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
        self.HandWriting.scribbleArea.showImage()

    def penColor(self):
        newColor = QColorDialog.getColor(self.HandWriting.scribbleArea.penColor())
        if newColor.isValid():
            self.HandWritingTab.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QInputDialog.getInt(self, "MNIST GUI",
                                           "Select pen width:",
                                           self.HandWriting.scribbleArea.penWidth(),
                                           1, 200, 1)
        if ok:
            self.HandWriting.scribbleArea.setPenWidth(newWidth)

    def about(self):
        QMessageBox.about(self, "About MNIST GUI",
                          "<p>The <b>MNIST GUI</b> provides hand-drawing tests.")

    def createActions(self):
        # Windows で表示された画像を閉じようとするとプログラムが終了してしまう。
        # 解決できるまでコメントアウト
        # self.showAct = QAction("&showImage", self,
        #                       triggered=self.showImage)

        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               triggered=self.close)

        self.penColorAct = QAction("&Pen Color...", self,
                                   triggered=self.penColor)

        self.penWidthAct = QAction("Pen &Width...", self,
                                   triggered=self.penWidth)

        self.clearScreenAct = QAction("&Clear Screen", self, shortcut="Space",
                                      triggered=self.HandWriting.scribbleArea.clearImage)

        self.aboutAct = QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QAction("About &Qt", self,
                                  triggered=QApplication.instance().aboutQt)

    def createMenus(self):
        fileMenu = QMenu("&File", self)
        # fileMenu.addAction(self.showAct)
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
        ret = QMessageBox.warning(self, "MNIST GUI",
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
