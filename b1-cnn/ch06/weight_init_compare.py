import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

weight_init_types = {"std=0.01": 0.01, "Xavier": "sigmoid", "He": "relu"}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    # 실험용 5층 레이어 - activation 옵션을 안주면 기본값은 relu
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_type,
    )
    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        # gradient 내에서도 loss를 계산하는데...아래 중복 코드인데...
        grads = networks[key].gradient(x_batch, t_batch)
        # 꼭 순전파를 다 계산해야 역전파가 계산되는 건 아니다...
        # 그냥 역전파 계산해서 손실값을 줄여나가는게 머신러닝이다...
        # 근데 생각해보니 손실값은 순전파 계산이네...결국 순전파 역전파 한 번씩은 계산해야 되네...
        optimizer.update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            # 이것도 위에서 계산 한 걸 중복해서...코드 편리성 때문인 듯...
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

# 그래프 그리기
markers = {"std=0.01": "o", "Xavier": "s", "He": "D"}
x = np.arange(max_iterations)
plt.figure(figsize=(8, 6))
for key in weight_init_types.keys():
    plt.plot(
        x,
        smooth_curve(train_loss[key]),
        marker=markers[key],
        markevery=100,
        label=key,
    )
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
