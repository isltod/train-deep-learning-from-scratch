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


# 이게 params나 grads에 같은 행렬 있으면 찾아서 지우는 코드라는데...
# 이게 왜 필요한지도 모르겠고...
def remove_duplicate(params, grads):
    # 일단 문제 안생기도록 복사해놓고 시작
    params, grads = params[:], grads[:]  # copy list

    # 아래서 for문으로 다 도는데 while True가 왜 필요하지?
    while True:
        find_flg = False
        L = len(params)

        # 뭔가 반복 없이 각 행렬별로 순서쌍을 만들려는 것 같은데...이게 아닐텐데...
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유, 즉 param내 어떤 행렬이 다른 행렬과 똑같을 때...
                if params[i] is params[j]:
                    # 같은 행렬이 나왔다면...경사를 더해? 같은 행렬을 제외하는 건 그렇다치고...
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)라...이건 뭔가?
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    # 배열이 몽땅 같다면 True, 아니면 False
                    and np.all(params[i].T == params[j])
                ):
                    # 위와 마찬가지로 경사는 더하고 같은 행렬은 제외...
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                # 같은 행렬 발견했으면 일단 i j 돌면서 찾는 걸 중지한다...밑에서 다시 돌아오게 해놨다...
                if find_flg:
                    break
            if find_flg:
                break

        # 뭔가 같은 행렬이 있었다면 다시 처음부터 찾아보고, 다 돌았는데도 같은 행렬 없다면 나가기
        if not find_flg:
            break
    return params, grads
