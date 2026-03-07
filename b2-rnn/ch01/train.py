import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.optimizer import SGD
from common.trainer import Trainer
from two_layer_net import TwoLayerNet
from dataset import spiral

# 하이퍼파라미터
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 데이터 로드
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
trainer.plot()
