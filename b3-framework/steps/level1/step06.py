import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        # 64쪽 그림 5-5에서 순전파 값은 data, 역전파 값은 grad
        self.grad = None


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 아래 역전파에서 도함수 자체가 아니라,
        # 현재 이 지점 input에서의 구체적인 기울기 값이 필요하므로 여기서 저장해놓는다...
        self.input = input
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.input.data
        # 역전파란 여기서 도함수 2x가 아니라 구체적으로 2 * 0.3 (input) 같은 값이 필요하다...
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        # 여기서도 도함수는 np.exp(), 그 구체적인 값은 거기에 순전파 때 실제 값 x를 넣어야 한다...
        gx = np.exp(x) * gy
        return gx


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


if __name__ == "__main__":
    # 순전파
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    print(y.data)

    # 역전파
    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
