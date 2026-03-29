import contextlib
import numpy as np
import weakref


class Config:
    # self.이 아니므로 클래스 속성 - static
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"{type(data)}은(는) 지원하지 않습니다. numpy.ndarray를 입력해주세요."
                )

        self.data = data
        self.name = name
        # 64쪽 그림 5-5에서 순전파 값은 data, 역전파 값은 grad
        self.grad = None
        self.creator = None
        # 역전파 자동 순서의 우선순위를 위한 세데인데...일단 0으로 초기화...
        self.generation = 0

    # # a*b에서 a의 __mul__이 호출되고, a는 self, b는 other
    # def __mul__(self, other):
    #     return mul(self, other)

    # def __add__(self, other):
    #     return add(self, other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"Variable({p})"

    def set_creator(self, func):
        self.creator = func
        # 함수와 변수의 generation이 상호작용한다는건데...
        self.generation = func.generation + 1

    # 재귀 호출 방식이 간단명료해보였는데, 뭔 문제가 있어서 바꾸는 모양인데...
    def backward(self, retain_grad=False):
        # 굳이 다 아는, 역전파 첫번째는 1이라는 것을 코드마다 넣지 않아도 되게 만드는...
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # generation으로 우선순위를 가지고 꺼내기 위한 조치들...
        funcs = []
        # 한 함수에서 둘 이상 변수가 출력되면, 변수마다 함수를 등록하려고 할 때 중복이 발생하고...
        # 함수 리스트에서 그걸 꺼내면 같은 backward가 여러 번 실행될 수 있다..그래서 set
        seen_set = set()

        # 요 아래서만 쓴다면 def 필요없지만,
        # while문 마지막에 새로 생긴 변수에 대해서도 같은 짓을 해야 하므로 함수로 만들어둔다...
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        # 1. 일단 내 바로 위 함수 하나만 가지고 리스트를 만들겠지?
        add_func(self.creator)

        while funcs:
            # 그걸 꺼내서 f에 넣고, 리스트는 빈 상태일테고...
            f = funcs.pop()

            # 출력 변수 미분 -> 함수 미분 -> 입력 변수 미분으로 역전파 전달하는데...
            # 우선 뒤인 출력 변수에 있는 미분
            # weakref는 바로 참조가 안되고 함수처럼 ()로 부른 후에 .grad 호출이 되는 모양...
            gys = [output().grad for output in f.outputs]
            # 함수 미분을 일단 담아놓고
            gxs = f.backward(*gys)
            # 이게 튜플이거나 스칼라니까, 일단 튜플로 통일해서 앞의 입력 변수 미분으로 가자...
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # 입력 변수와 전달할 미분의 갯수는 같을테니 묶는데 zip이면 순서도 맞게 되는 건가?
            for x, gx in zip(f.inputs, gxs):
                # 같은 변수를 여러번 입력하는 경우는 분기가 된다. 즉 x -> x, x -> add(x, x)니까...
                # 이 경우 add의 미분을 일단 구하고, 그걸 더해야한다...
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                # 그리고 앞에 또 함수가 있는지는 각 변수들마다 확인하니, for 문으로 들어온다...
                if x.creator is not None:
                    # funcs.append(x.creator) 대신 위 add_func를 쓰면 집합을 이용해서 추가하고 정렬해서 꺼낸다...
                    add_func(x.creator)
            # 이 while 사이클에서 func를 중심으로, outputs에서 grad를 gys로 받고, 그걸 inputs에 gxs로 전달했다..
            # 그럼 이제 이 func의 outputs에 대해서는 볼일 끝났고, 그래서 retain 옵션 없으면 grad 삭제...
            # 말단 xs는 어떤 func의 outputs일 수가 없으니 자동으로 grads가 살아남는다...
            if not retain_grad:
                for y in f.outputs:
                    # outputs는 순환참조 문제 때문에 약한 참조 상태이고, 그래서 ().grad
                    y().grad = None

    def cleargrad(self):
        self.grad = None


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    # inputs에 * 붙이면 입력을 리스트로 묶을 필요 없이 그냥 주욱 나열하면 된다...그걸 여기서 묶는다..
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        # 이게 Variable이 아니라 np.ndarray일 경우에 대비해서 위에...
        xs = [x.data for x in inputs]
        # 언패킹...
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # train에서는 역전파를 위해 generation과 inputs 보관, predict에는 필요 없으니 안하기...
        if Config.enable_backprop:
            # 함수의 세대는 입력 중 max를 쓰는데, 그 값을 output에 전달해야 하니, creator로 넣기 전에 설정한다...
            self.generation = max([x.generation for x in inputs])
            # 이게 함수와 변수를 연결...
            for output in outputs:
                output.set_creator(self)
            # 아래 역전파에서 도함수 자체가 아니라,
            # 현재 이 지점 input에서의 구체적인 기울기 값이 필요하므로 여기서 저장해놓는다...
            # input이 함수를 참조하지 않고 함수만 input을 참조하므로 순환참조가 아니고, 따라서 weakref 필요없다...
            self.inputs = inputs
            # 약한 참조는 weakref.ref(outputs)처럼 한 번에 하면 안되고, 원소별로 따로따로 지정해야 하나?
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# Function 클래스는 좀 더 복잡해졌지만, 덕분에 여기부터 나머지 코드들은 더 쉬워진다...
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


# 2-a 경우 rsub에서는 a가 첫 번째 x0의 자리로, 2가 x1의 자리로 들어간다...
# 여기서 2 입장에서 sub가 호출되지 않고 a의 rsub가 호출되는 이유는 __array_priority__ = 200 이거 때문...
def rsub(x0, x1):
    # 그래서 스칼라 2인 x1을 array로 바꾸고
    x1 = as_array(x1)
    # 이걸 정상적인 Sub로 보낼 때는 순서를 바꿔 보내야 한다...
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


# 이 경우도 2/a에서 a가 x0, 2가 x1으로 들어오니까
def rdiv(x0, x1):
    # 2인 x1을 array로 만들고
    x1 = as_array(x1)
    # 실제 나누기를 할 때는 순서를 바꿔 넣어야 한다...
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = sub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

if __name__ == "__main__":
    a = Variable(np.array([1.0]))
    b = np.array([2.0])
    c = b + a
    print(c)
    print(c.data)
    x = Variable(np.array(2.0))
    y = -x
    print(y)
    y1 = 2.0 - x
    print(y1)
    y2 = x - 1.0
    print(y2)
    y = x**3.0
    print(y)
    y
