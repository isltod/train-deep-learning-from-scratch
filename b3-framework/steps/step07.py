import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        # 64쪽 그림 5-5에서 순전파 값은 data, 역전파 값은 grad
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 1. creater로 함수 찾고, 2. input으로 변수 찾고, 3. backward에 먹이기...
    def backward(self):
        # 1. 나를 만든 함수를 찾고
        f = self.creator
        if f is not None:
            # 2. 그 앞에 변수를 찾고(여기에 grad 전파가 역전파)
            x = f.input
            # 3. 찾은 함수의 backward에 내 grad 먹여서 역전파...
            x.grad = f.backward(self.grad)
            # 4. 그리고 그 변수의 역전파를 또 부르면...재귀 호출로 계속...
            x.backward()


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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

    # 이제 간단하게 역전파...
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
