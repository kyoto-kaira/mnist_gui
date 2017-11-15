from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import threading
import datetime
import pickle

from model_creator import ModelCreator


class RankingData:
    def __init__(self):
        try:
            f = open("./ranking.pickle", "rb")
            self._data = pickle.load(f)
        except:
            self._data = list()

    def insert(self, name):
        item = dict()
        item.update({"name": name,
                     "model_creator": None,
                     "f1-score": None,
                     "model_file_name": None,
                     "password": None})
        self._data.append(item)
        print(self._data)

        d = datetime.datetime.today()
        print(name + "_" + d.strftime("%Y-%m-%d_%H-%M-%S"))

        with open("./ranking.pickle", "wb") as f:
            pickle.dump(self._data, f)


class RankingRegisterDialog(QDialog):
    def __init__(self, parent=None):
        super(RankingRegisterDialog, self).__init__(parent)
        self.setWindowTitle("登録")

        label_name = QLabel("名前")
        self.input_name = QLineEdit()
        re = QRegularExpression("[^/\\\\ !\"#$%&'()\\-=\\^~|@`[{;+:*\\]}?<>]+")
        self.input_name.setValidator(QRegularExpressionValidator(re))
        layout_name = QHBoxLayout()
        layout_name.addWidget(label_name)
        layout_name.addWidget(self.input_name)

        label_score = QLabel("スコア")
        self.label_score_value = QLabel("0.99")
        layout_score = QHBoxLayout()
        layout_score.addWidget(label_score)
        layout_score.addWidget(self.label_score_value)

        register_btn = QPushButton("ランキングに登録")
        register_btn.clicked.connect(self.register)

        layout = QVBoxLayout()
        layout.addLayout(layout_name)
        layout.addLayout(layout_score)
        layout.addWidget(register_btn)

        self.setLayout(layout)
        self.setFixedSize(300, 100)

        self.register_func = None

    def show_dialog(self):
        self.input_name.setText("")
        self.label_score_value.setText("計算中...")
        self.show()
        tr = threading.Thread(target=self._calc_score)
        tr.start()

    def _calc_score(self):
        import time
        time.sleep(1)
        self.label_score_value.setText("0.99")

    def register(self):
        name = self.input_name.text()
        if len(name) is 0:
            QMessageBox.warning(self, "メッセージ",
                                "名前を入力してください",
                                QMessageBox.Ok)
            return
        if self.register_func is not None:
            self.register_func(name)
        else:
            print(name)
        self.close()

    def set_register_func(self, func):
        self.register_func = func


class RankingWidget(QWidget):
    def __init__(self, model, parent=None):
        super(RankingWidget, self).__init__()

        self.ranking_data = RankingData()

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0x2a, 0x2a, 0x2a))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.register_dialog = RankingRegisterDialog(self)
        self.register_dialog.set_register_func(self.ranking_data.insert)

        self.register_btn = QPushButton("登録", self)
        self.register_btn.clicked.connect(self.register_dialog.show_dialog)

        self.ranking_table = QTableWidget(self)
        self.ranking_table.setColumnCount(2)
        self.ranking_table.setHorizontalHeaderLabels(["名前", "スコア"])
        self.ranking_table.setRowCount(1000)
        for i in range(1000):
            item_name = QTableWidgetItem("名前" + str(i))
            item_score = QTableWidgetItem("0.99")
            self.ranking_table.setItem(i, 0, item_name)
            self.ranking_table.setItem(i, 1, item_score)

    def resizeEvent(self, QResizeEvent):
        self.register_btn.move(self.width() * 0.1, self.height() * 0.1)

        self.ranking_table.move(self.width() * 0.3, self.height() * 0.1)
        self.ranking_table.resize(self.width() * 0.6, self.height() * 0.8)
