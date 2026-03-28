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
        # 역전파 자동 순서의 우선순위를 위한 세데인데...일단 0으로 초기화...
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        # 함수와 변수의 generation이 상호작용한다는건데...
        self.generation = func.generation + 1

    # 재귀 호출 방식이 간단명료해보였는데, 뭔 문제가 있어서 바꾸는 모양인데...
    def backward(self):
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
            gys = [output.grad for output in f.outputs]
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

    def cleargrad(self):
        self.grad = None


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
        # 함수의 세대는 입력 중 max를 쓰는데, 그 값을 output에 전달해야 하니, creator로 넣기 전에 설정한다...
        self.generation = max([x.generation for x in inputs])
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


# Function 클래스는 좀 더 복잡해졌지만, 덕분에 여기부터 나머지 코드들은 더 쉬워진다...
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x**2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    b = square(a)
    c = square(a)
    y = add(b, c)
    y.backward()
    print(y.data)
    print(x.grad)
    print(a.grad)
    print(b.grad)
    print(c.grad)
