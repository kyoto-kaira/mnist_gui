from PyQt5.QtGui import QTextCursor

import numpy as np
import threading
import copy
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import load_model, model_from_json
from keras.callbacks import LambdaCallback

import sklearn.metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split

default_model_path = './model.hdf5'


class MnistModel(threading.Thread):
    def __init__(self, logger, progress):
        super(MnistModel, self).__init__()
        self.logger = logger
        self.progress = progress

        self.learn_event = threading.Event()
        self.learn_event.clear()
        self._exit = False
        self._is_learning = False

        self.mnist = datasets.fetch_mldata('MNIST original', data_home='.')
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self._set_train_and_test_data()

        self.model = None
        try:
            # self.load(default_model_path)

            # load json and create model
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model.h5")
            self.model = loaded_model
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(lr=0.01),
                               metrics=['accuracy'])
            print("Loaded model from disk")

            """
            # serialize model to JSON
            model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights("model.h5")
            """
            self.model_creator = None
        except:
            self.set_model()
        self.graph = tf.get_default_graph()

        self.update_bar_func = None
        self.model_creator = None

    def _set_train_and_test_data(self):
        np.random.seed(0)
        n = len(self.mnist.data)
        indices = np.random.permutation(range(n))[:n]

        x = self.mnist.data[indices]
        x = x.reshape(-1, 28, 28, 1)
        y = self.mnist.target[indices]
        y = np.eye(10)[y.astype(int)]  # 1-of-K 表現に変換

        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(x, y, test_size=0.2)

    def load(self, path):
        if self._is_learning:
            raise RuntimeError("学習中なのでモデルのロードはできません。")
        else:
            self.model = load_model(path)
        self.model_creator = None

    def save(self, path):
        if self._is_learning:
            raise RuntimeError("学習中なので、モデルはセーブできません。")
        else:
            self.model.save(path)

    def set_update_bar_func(self, update_bar_func):
        self.update_bar_func = update_bar_func

    def run(self):
        """学習を実行する。時間がかかるのでマルチスレッド化してある。"""
        while True:
            self.learn_event.wait()
            if self._exit:
                break
            self._is_learning = True
            epochs = 1
            batch_size = 1000
            num_batch = len(self.X_train) // batch_size
            if self.model is None:
                self.logger.append("no model")
                return

            # self.progress.setValue(0)
            self.logger.append("start learning")

            def batch_end_out(epoch, logs):
                # self.progress.setValue((epoch + 1) / num_batch * 100)
                self.logger.append(str("{}/{} {:.4f}".format(epoch + 1,
                                                             num_batch,
                                                             logs['acc'])))
                self.logger.moveCursor(QTextCursor.End)
                if self.update_bar_func is not None:
                    self.update_bar_func()

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

            self._is_learning = False
            if self._exit:
                break
            self.learn_event.clear()
            if self._exit:
                break

    def is_learning(self):
        return self._is_learning

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

    def set_model(self, model=None, model_creator=None):
        if self._is_learning:
            raise RuntimeError("学習中なので、モデルの設定はできません。")
        if model is None:
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
            # serialize model to JSON
            model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights("model.h5")
            self.model_creator = None
        else:
            self.model = model
            self.model_creator = copy.copy(model_creator)

    def report_evaluation(self):
        with self.graph.as_default():
            y = self.model.predict(self.X_test)
            y_pred = [np.argmax(onehot) for onehot in y]
            y_true = [np.argmax(onehot) for onehot in self.Y_test]
            # return sklearn.metrics.classification_report(y_true, y_pred)
            return float(sklearn.metrics.f1_score(y_true, y_pred, average='weighted'))
