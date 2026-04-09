from dezero import cuda
from dezero import Parameter
from dezero import utils
from dezero.utils import pair
import dezero.functions as F
import numpy as np
import os
import weakref


class Layer:
    def __init__(self):
        # 집합 형식의 인스턴스 변수...
        self._params = set()

    # 인스턴스 변수를 설정할 때 호출되는 메서드...
    def __setattr__(self, name, value):
        # 값이 매개변수 또는 레이어일 때, 값이 아니라 이름을 저장한다...그럼 그걸로 dict가 된다...
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # 여긴 Functions와 다르게 input, output 다 약한 참조...
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                # yield는 작업 종료 없이 return, 그래서 for 문 등에서 쓰는구만...
                # yield를 사용하는 함수를 제너레이터라고 한다고...
                # yield from은 다른 제너레이터나 반복 객체에서 하나씩 꺼내 반환
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    # params_dict라는 걸 받아서, 거기에 키를 만들고 현재 __dict__를 넣는다?
    def _flatten_params(self, params_dict, parent_key=""):
        # 레이어, 가중치들 이름으로 돌면서...
        for name in self._params:
            # 만든 인스턴스 변수들이 담겨있는 사전...
            obj = self.__dict__[name]
            key = parent_key + "/" + name if parent_key else name

            # 레이어면 다시 재귀호출로, 아니고 가중치면 그 값을 사전에 담기...
            # 사전은 참조변수로 밖에서 이용...
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        # 파일은 임시 저장소에 넣자...
        if not os.path.exists(utils.TMP_DIR):
            os.mkdir(utils.TMP_DIR)
        path = os.path.join(utils.TMP_DIR, path)

        # 먼저 cupy면 numpy 변수로 변경하고 저장
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        # 매개변수는 Parameter 클래스, 넘파이 배열은 그 data
        array_dict = {
            key: param.data for key, param in params_dict.items() if param is not None
        }

        # KeyboardInterrupt는 Ctrl+C 같은 키로 중단시킬 때...
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            # 쓰다 만 데이터는 지운다...
            if os.path.exists(path):
                os.remove(path)
            raise

    # 레이어부터 쌓아주는 것이 아니라,
    # 기본 레이어 구조는 다 만들어놓고, 거기에 매개변수만 저장 값으로 채우기...
    def load_weights(self, path):
        # # 파일은 항상 임시 디렉토리에서 읽자...없으면 그냥 돌아가기
        # path = os.path.join(utils.TMP_DIR, path)
        # if not os.path.exists(path):
        #     print("No file or directory for Layer.load_wegiths: '{}'.".format(path))
        #     return

        npz = np.load(path)
        params_dict = {}
        # 그래서 현재 레이어 구조의 매개변수들을 참조 변수로 가져오고
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            # 그 매개변수들에 같은 키의 저장값을 넣어주기...
            param.data = npz[key]


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        # in_size 있으면 init에서 가중치 초기화하고...
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        # 이거 왜 앞에는 xp고 뒤에는 굳이 np지? xp가 맞는거 아닌가? 에러날라나?
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # in_size 없다면 입력 데이터 확인해서 forward에서 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        pad=0,
        nobias=False,
        dtype=np.float32,
        in_channels=None,
    ):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(
        self,
        out_channels,
        kernel_size,
        stride=1,
        pad=0,
        nobias=False,
        dtype=np.float32,
        in_channels=None,
    ):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y
