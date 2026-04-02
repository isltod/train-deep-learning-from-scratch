import numpy as np

x = np.array(1)
print(id(x))
# 누적 대입 연산자는 값만 덮어쓰는 인플레이스 연산 - id가 같다...
x += x
print(id(x))
# 이건 id가 달라진다..
x = x + x
print(id(x))


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"{type(data)}은(는) 지원하지 않습니다. numpy.ndarray를 입력해주세요."
                )

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()

            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                    # 이렇게 쓰면 인플레이스 연산, 즉 id(gx) = id(x.grad)이고,
                    # 원래 gx는 gy에서 왔으므로 결국 id(y.grad) = id(x.grad)가 되버린다...
                    # x.grad += gx
                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    # inputs에 * 붙이면 입력을 리스트로 묶을 필요 없이 그냥 주욱 나열하면 된다...그걸 여기서 묶는다..
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        # 언패킹...
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # 이게 함수와 변수를 연결...
        for output in outputs:
            output.set_creator(self)
        # 아래 역전파에서 도함수 자체가 아니라,
        # 현재 이 지점 input에서의 구체적인 기울기 값이 필요하므로 여기서 저장해놓는다...
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print("y.grad: {}({})".format(y.grad, id(y.grad)))
print("x.grad: {}({})".format(x.grad, id(x.grad)))
