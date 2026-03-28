import numpy as np


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
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        # 이게 함수와 변수를 연결...
        for output in outputs:
            output.set_creator(self)
        # 아래 역전파에서 도함수 자체가 아니라,
        # 현재 이 지점 input에서의 구체적인 기울기 값이 필요하므로 여기서 저장해놓는다...
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)


if __name__ == "__main__":
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)
