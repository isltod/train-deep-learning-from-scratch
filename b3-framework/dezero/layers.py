from dezero import Parameter
import dezero.functions as F
import numpy as np
import weakref


class Layer:
    def __init__(self):
        # 집합 형식의 인스턴스 변수...
        self._params = set()

    # 인스턴스 변수를 설정할 때 호출되는 메서드...
    def __setattr__(self, name, value):
        # 값이 매개변수일 때, 값이 아니라 이름을 저장한다...그럼 그걸로 dict가 된다...
        if isinstance(value, Parameter):
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
            # yield는 작업 종료 없이 return, 그래서 for 문 등에서 쓰는구만...
            yield getattr(self, name)

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


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

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # in_size 없다면 입력 데이터 확인해서 forward에서 가중치 초기화
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = F.linear(x, self.W, self.b)
        return y
