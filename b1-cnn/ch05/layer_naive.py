class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        # 위에서 받은 값에 x, y를 바꿔서 곱해서 보낸다...
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        # 곱셈과 달리 역전파에서 가지고 있던 값을 곱해줄 필요가 없으니 인스턴스 변수 없음...
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
