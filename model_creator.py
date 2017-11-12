from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D


class LayerBase:
    def __init__(self):
        self.output_shape = ()

    def get_output_shape(self):
        return self.output_shape

    def get_code(self):
        return ""


class ActivationLayer(LayerBase):
    def __init__(self, input_shape, func_name):
        if not self.do_support(func_name):
            raise RuntimeError("関数 {} はサポートしていません。"
                               .format(func_name))
        self.input_shape = input_shape
        self.func_name = func_name
        self.output_shape = input_shape

    def get_code(self):
        return "model.add(Activation('{}', input_shape={})); "\
               .format(self.func_name,
                       str(self.input_shape))

    @staticmethod
    def get_func_set():
        return {'relu', 'sigmoid', 'softmax'}

    @staticmethod
    def do_support(str):
        return str in ActivationLayer.get_func_set()


class DenseLayer(LayerBase):
    def __init__(self, input_shape, units):
        if len(input_shape) is not 1:
            raise RuntimeError("全結合層の前の input_shape は１次元である必要があります。")
        self.input_shape = input_shape
        self.units = units
        self.output_shape = (units,)

    def get_code(self):
        return "model.add(Dense({}, input_shape={})); "\
               .format(self.units,
                       str(self.input_shape))

class FlattenLayer(LayerBase):
    def __init__(self, input_shape):
        dim = 1
        for v in input_shape:
            dim *= v
        self.input_shape = input_shape
        self.output_shape = (dim,)

    def get_code(self):
        return "model.add(Flatten(input_shape={})); " \
               .format(str(self.input_shape))


class Conv2dLayer(LayerBase):
    def __init__(self, input_shape, filters, kernel_x, kernel_y):
        if len(input_shape) is not 3:
            raise RuntimeError("畳み込み層の前の input_shape は３次元である必要があります。")
        output_x = input_shape[0] - kernel_x + 1
        output_y = input_shape[1] - kernel_y + 1
        if output_x < 1:
            raise RuntimeError("カーネルサイズ(x)が大きすぎます。"
                               "{}以下を指定してください。"
                               .format(input_shape[0]))
        if output_y < 1:
            raise RuntimeError("カーネルサイズ(y)が大きすぎます。"
                               "{}以下を指定してください。"
                               .format(input_shape[1]))
        if filters < 1:
            raise RuntimeError("フィルターの数が小さすぎます。1以上を指定してください。")
        self.input_shape = input_shape
        self.filters = filters
        self.kernel = (kernel_x, kernel_y)
        self.output_shape = (output_x, output_y, filters)

    def get_code(self):
        return "model.add(Conv2D({}, {}, input_shape={})); " \
            .format(self.filters,
                    str(self.kernel),
                    str(self.input_shape))


class CompileLayer(LayerBase):
    def __init__(self, input_shape):
        if (len(input_shape) is not 1) or (input_shape[0] is not 10):
            raise RuntimeError("最終層は、(10,)の形である必要があります。")
        self.input_shape = input_shape
        self.output_shape = input_shape

    def get_code(self):
        return "model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc']);"


class ModelCreator(object):
    '''
    MNIST で使うモデルの構築をするクラス

    モデルの編集を関数として提供する。
    keras のモデルとして、出力する。
    input_shape=(28, 28, 1)
    output_shape=(10)
    となるようなモデルを構築する。
    '''
    def __init__(self):
        self.changed_notify_func = None

        self.shape = None
        self.model_structure = None
        self.valid = None
        self.is_last_layer_softmax = None
        self.clear()
        pass

    def __iter__(self):
        return iter(self.model_structure)

    def clear(self):
        self.shape = (28, 28, 1)
        self.model_structure = []
        self.valid = False
        self.is_last_layer_softmax = False
        self.call_notify_func()

    def _add_layer(self, layer):
        self.model_structure.append(layer)
        self.shape = layer.output_shape
        self.is_last_layer_softmax = False
        self.call_notify_func()

    def add_activation(self, func_name):
        self._add_layer(ActivationLayer(self.shape, func_name))
        if func_name == "softmax":
            self.is_last_layer_softmax = True

    def add_dense(self, units):
        self._add_layer(DenseLayer(self.shape, units))

    def add_flatten(self):
        self._add_layer(FlattenLayer(self.shape))

    def add_conv2d(self, filters, kernel_x, kernel_y):
        self._add_layer(Conv2dLayer(self.shape, filters, kernel_x, kernel_y))

    def add_compile(self):
        layer = CompileLayer(self.shape)
        if not self.is_last_layer_softmax:
            self.add_activation("softmax")
            print("最終層に softmax の層を追加しました。")
        self._add_layer(layer)
        self.valid = True

    def get_model(self):
        if not self.valid:
            raise RuntimeError("モデルが正しくありません。モデルを修正してください。")
        code = ""
        for layer in self.model_structure:
            code += layer.get_code()
        print(code)
        model = Sequential()
        exec(code)
        model.summary()
        return model

    def set_changed_notify(self, func):
        self.changed_notify_func = func

    def call_notify_func(self):
        if self.changed_notify_func is not None:
            self.changed_notify_func()
