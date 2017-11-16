from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import threading
import datetime
import pickle
from one_line_info import *


class RankingData:
    def __init__(self):
        try:
            f = open("./ranking_data/ranking.pickle", "rb")
            self._data = pickle.load(f)
        except:
            self._data = list()

        self.update_notify_func = None

    def insert(self, name: str, f1_score: float, mnist_model):
        item = dict()
        d = datetime.datetime.today()
        model_file_name = "./ranking_data/"\
                          + name + "_"\
                          + d.strftime("%Y-%m-%d_%H-%M-%S")\
                          + ".hdf5"
        item.update({"name": name,
                     "f1-score": f1_score,
                     "model_file_name": model_file_name,
                     "model_creator": mnist_model.model_creator,
                     })
        self._data.append(item)
        print(self._data)

        if self.update_notify_func is not None:
            self.update_notify_func()

        try:
            mnist_model.save(model_file_name)
        except:
            print(model_file_name + "は保存されませんでした。")

        with open("./ranking_data/ranking.pickle", "wb") as f:
            pickle.dump(self._data, f)

    def get_sorted_data(self):
        return sorted(self._data, key=lambda item: - item['f1-score'])

    def set_update_notify_func(self, func):
        self.update_notify_func = func


class RankingRegisterDialog(QDialog):
    def __init__(self, mnist_model, parent=None):
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
        self.label_score_value = QLabel("")
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
        self.setFixedSize(300, 200)

        self.mnist_model = mnist_model
        self.register_func = None

        self.score_calculated = False
        self.score = 0.0

    def show_dialog(self):
        if self.mnist_model.model_creator is None:
            global_one_line_info.send("エディターで作られたモデルではありません。")
            return
        self.input_name.setText("")
        self.label_score_value.setText("計算中...")
        self.score_calculated = False
        self.show()
        tr = threading.Thread(target=self._calc_score)
        tr.start()

    def _calc_score(self):
        score = self.mnist_model.report_evaluation()
        self.label_score_value.setText(str(score))
        self.score = score
        self.score_calculated = True

    def register(self):
        name = self.input_name.text()
        if len(name) is 0:
            QMessageBox.warning(self, "メッセージ",
                                "名前を入力してください",
                                QMessageBox.Ok)
            return
        if not self.score_calculated:
            QMessageBox.warning(self, "メッセージ",
                                "スコアを計算中です",
                                QMessageBox.Ok)
            return
        if self.mnist_model.model_creator is None:
            QMessageBox.warning(self, "メッセージ",
                                "エディターで作られたモデルではありません。",
                                QMessageBox.Ok)
            return
        if self.register_func is not None:
            self.register_func(name, self.score, self.mnist_model)
        else:
            print(name)
        self.close()

    def set_register_func(self, func):
        self.register_func = func


class RankingWidget(QWidget):
    def __init__(self, mnist_model, parent=None):
        super(RankingWidget, self).__init__()

        self.ranking_data = RankingData()

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0x2a, 0x2a, 0x2a))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        self.register_dialog = RankingRegisterDialog(mnist_model, self)
        self.register_dialog.set_register_func(self.ranking_data.insert)

        self.register_btn = QPushButton("登録", self)
        self.register_btn.clicked.connect(self.register_dialog.show_dialog)

        self.ranking_table = QTableWidget(self)
        self.ranking_table.setColumnCount(2)
        self.ranking_table.setHorizontalHeaderLabels(["名前", "スコア"])

        self.update_ranking()
        self.ranking_data.set_update_notify_func(self.update_ranking)

    def resizeEvent(self, QResizeEvent):
        self.register_btn.move(self.width() * 0.1, self.height() * 0.1)

        self.ranking_table.move(self.width() * 0.3, self.height() * 0.1)
        self.ranking_table.resize(self.width() * 0.6, self.height() * 0.8)

    def update_ranking(self):
        ranking = self.ranking_data.get_sorted_data()
        self.ranking_table.setRowCount(len(ranking))
        for i, item in enumerate(ranking):
            item_name = QTableWidgetItem(item['name'])
            item_name.setFlags(Qt.ItemIsEnabled)
            item_score = QTableWidgetItem(str(item['f1-score']))
            item_score.setFlags(Qt.ItemIsEnabled)
            self.ranking_table.setItem(i, 0, item_name)
            self.ranking_table.setItem(i, 1, item_score)
