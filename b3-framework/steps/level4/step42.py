import sys

sys.path.append("b3-framework")

from dezero import Variable
import dezero.functions as F
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
# (100x1) 난수 배열
x = np.random.rand(100, 1)
# 거기에 선형 관계 + 잔차 더하기
# y = 5 + 2 * x
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    # 여기서 b는 브로드캐스트 발생
    y = F.matmul(x, W) + b
    return y


# 이건 계산 그래프를 복잡하게 만들어 중간 변수를 많이 만드니,
# 메모리 효율을 위해 functions에 만든 MES를 사용...
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    # 매개변수 갱신은 lr * W가 아니로 W.data에 대해서 하는데...
    # 이렇게 하면 넘파이 곱이 되고 계산 그래프를 만들지 않는다...
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)

# s는 마커 사이즈...
plt.scatter(x.data, y.data, s=10)
plt.xlabel("x")
plt.ylabel("y")
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color="r")
plt.tight_layout()
plt.show()
