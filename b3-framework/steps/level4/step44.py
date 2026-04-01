import sys

sys.path.append("b3-framework")

from dezero import Variable, Parameter
import dezero.functions as F
import dezero.layers as L
import numpy as np
import matplotlib.pyplot as plt

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))
y = x * p

print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))

layer = L.Layer()

layer.p1 = Parameter(np.array(2.0))
layer.p2 = Parameter(np.array(3.0))
# 이렇게 하면 p3, p4가 저장 안되는데...
layer.p3 = Variable(np.array(4.0))
layer.p4 = "test"

print(layer._params)
print("----------------------")
for name in layer._params:
    # 여기서 왜 결과가 Variable()로 나오는 거지?
    print(name, layer.__dict__[name])

# step43 문제를 새로 만든 Layer 클래스를 이용해서 접근...
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# out_size만 지정...in_size는 입력 데이터 보고 결정하고, 편향은 없고, 데이터 타입은 np.float32
l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    # Layer 클래스를 Function 클래스처럼 __call__() -> forward() 구조로 만들었으므로,
    # 그냥 이렇게 부르면 자동으로 forward 계산되고 그래프 생성...
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            # 가중치는 가중치의 미분으로 갱신한다...
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0.0, 1.0, 0.01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color="r")
plt.tight_layout()
plt.show()
