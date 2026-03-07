import numpy
import time
import matplotlib.pyplot as plt
from common.np import *
from common.util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # 뒤섞기
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                # 기울기는 반환값으로 받는게 아니라, 참조 변수로 model 안의 grads에 있는걸 수정...
                model.backward()
                params, grads = model.params, model.grads

                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # 가중치 수정도 반환값 아니라 참조 변수로...
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        "| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 손실 %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            avg_loss,
                        )
                    )
                    # 이렇게 하면 avg_loss가 cupy에서 cpu 실수로 변하나?
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1

    def plot(self, ylim=None):
        # 넘파이는 명시적으로 numpy, np는 GPU 설정에 따라 cupy일 수도...
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel("반복 (x" + str(self.eval_interval) + ")")
        plt.ylabel("손실")
        plt.show()
