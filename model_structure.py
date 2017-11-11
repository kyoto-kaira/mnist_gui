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
        return " model.add(Activation('{}', input_shape={}));"\
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
        return " model.add(Dense({}, input_shape={}));"\
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
        return " model.add(Flatten(input_shape={}));" \
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
        return " model.add(Conv2D({}, {}, input_shape={}));" \
            .format(self.filters,
                    str(self.kernel),
                    str(self.input_shape))


class InitialLayer(LayerBase):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def get_code(self):
        return "model = Sequential();"


class CompileLayer(LayerBase):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def get_code(self):
        return " model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc']);"


class ModelStructure(object):
    '''
    MNIST で使うモデルをリストや辞書を使って表現する。
    モデルの編集を関数として提供する。
    keras のモデルとして、出力する。
    input_shape=(28, 28, 1)
    output_shape=(10)
    となるようなモデルを構築する。
    '''
    def __init__(self):
        self.shape=(28,28,1)
        self.model_structure = []
        pass

    def _add_layer(self, layer):
        self.model_structure.append(layer)
        self.shape = layer.output_shape

    def add_activation(self, str):
        self._add_layer(ActivationLayer(self.shape, str))

    def add_dense(self, units):
        self._add_layer(DenseLayer(self.shape, units))

    def add_flatten(self):
        self._add_layer(FlattenLayer(self.shape))

    def add_conv2d(self, filters, kernel_x, kernel_y):
        self._add_layer(Conv2dLayer(self.shape, filters, kernel_x, kernel_y))

    def add_initial(self):
        self._add_layer(InitialLayer(self.shape))

    def add_compile(self):
        self._add_layer(CompileLayer(self.shape))

    def get_model(self):
        self.add_initial()
        self.add_conv2d(10, 3, 3)
        self.add_activation('relu')
        self.add_flatten()
        self.add_dense(100)
        self.add_compile()
        str = ""
        for layer in self.model_structure:
            str += layer.get_code()
        print(str)
        return str


m = ModelStructure()
code = m.get_model()
exec(code)
model.summary()
