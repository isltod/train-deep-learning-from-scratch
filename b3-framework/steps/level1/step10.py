import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f"{type(data)}은(는) 지원하지 않습니다. numpy.ndarray를 입력해주세요."
                )

        self.data = data
        # 64쪽 그림 5-5에서 순전파 값은 data, 역전파 값은 grad
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 재귀 호출 방식이 간단명료해보였는데, 뭔 문제가 있어서 바꾸는 모양인데...
    def backward(self):
        # 굳이 다 아는, 역전파 첫번째는 1이라는 것을 코드마다 넣지 않아도 되게 만드는...
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 1. 일단 내 바로 위 함수 하나만 가지고 리스트를 만들겠지?
        funcs = [self.creator]
        while funcs:
            # 그걸 꺼내서 f에 넣고, 리스트는 빈 상태일테고...
            f = funcs.pop()

            # 함수 중심으로 앞 뒤 변수 받아 역전파 계산해 넣고, 여긴 좀 똑 떨어져서 읽기 좋게 되네...
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            # 앞에 변수에 그 앞이 더 있다면 리스트에 추가...
            # 빈 리스트에 뭔가 들어가겠지...결국 재귀랑 비슷한데...
            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        # 이게 함수와 변수를 연결...
        output.set_creator(self)
        # 아래 역전파에서 도함수 자체가 아니라,
        # 현재 이 지점 input에서의 구체적인 기울기 값이 필요하므로 여기서 저장해놓는다...
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        # 근데 이게 np.float64를 반환해서 여전히 문제가 남아있네...그래서 위에 as_array 만들고 Function에 적용...
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


def square(x):
    # Square() 까지가 Square.__init__, 그 뒤의 (x)는 Function.call(x)->Square.forward(x)
    return Square()(x)


def exp(x):
    return Exp()(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        # Square의 data에 들어가는게 아니라, 제곱하고 그게 y로...
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


if __name__ == "__main__":
    unittest.main()
