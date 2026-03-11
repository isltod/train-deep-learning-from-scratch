import sys

sys.path.append("..")
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
            # 매 epoch마다 데이터 뒤섞기
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                # 각 epoch 내에서, 앞에서부터 batch_size 만큼씩 떠서 학습...
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                # 기울기는 반환값으로 받는게 아니라, 참조 변수로 model 안의 grads에 있는걸 수정...
                model.backward()
                # in_layers 10개(window 5 * 2) 중복 + ns_loss 6개(정답 1 + sample_size 5) 중복인데,
                # 여기서 대표 각 1개씩 뽑았는데, 이게 params - model.params[0~전부]와 다 같은 메모리(즉 같은 객체)...
                # 왜나면 처음 만들 때 W_in 하나를 만들어서 모든 레이어에 넣었고,
                # remove_dup에서도 얕은 복사로 메모리를 공유했으므로...
                # 그래서 대표로 뽑은 grads, params만 고쳐도 나머지(16개)가 싹다 고쳐진다...
                # 더구나 같은 놈이 나오면 대표에게 grad를 몰아준다...그래서 이게 있어야 학습이 되는 모양...
                params, grads = remove_duplicate(model.params, model.grads)
                # params, grads = model.params, model.grads

                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # 가중치 수정도 반환값 아니라 참조 변수로, grads를 가지고 params를 변경하는데...근데 이건 self.params 아닌데?
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


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        # 확률 역수 perplexity 목록이겠지...
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    # 전체 훈련 단어들과 정답지, 배치 크기, RNN 펼치기 수 = 단어 순서
    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")

        # 0~배치 크기 99까지, 배치 덩어리 크기로 점프 리스트...[0, 10, 20, ..., 980, 990]
        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for time in range(time_size):
            # 한 배치 내에서 반복
            for i, offset in enumerate(offsets):
                # 단어 순서 3, 배치 번호 4는 batch_x[4, 3]에, 그 다음 단어는 batch_t[4, 3]에 저장...
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            # 위에 time_size는 항상 5 고정, 즉 5번씩 끊어 읽는데...
            # 이걸 에포크에서 반복해서 한 배치는 10개지만 총 1000개의 데이터를 배치 100개로 읽어야 하니까,
            # 0~99까지 자리를 바꿔가며 읽기 위해서 time_idx를 1씩 증가
            self.time_idx += 1
        # 배치 N x 단어 순서 T 행렬
        return batch_x, batch_t

    # 훈련 데이터(처음부터 -1까지 문장), 정답지(2에서 마지막까지 문장), 100 에포크 돌기, 10개씩 끊어 처리, RNN 펼치기 5개
    def fit(
        self,
        xs,
        ts,
        max_epoch=10,
        batch_size=20,
        time_size=35,
        max_grad=None,
        eval_interval=20,
    ):
        # 훈련 문장 단어들 수
        data_size = len(xs)
        # 에포크 안에서 반복은 전체 훈련 단어 수를 배치 크기 * RNN 펼치기로 나누기
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                # 전체 훈련 단어들과 정답지, 배치 크기, RNN 펼치기 수 넣고,
                # 전체 단어 수를 배치 크기로 나눈 덩어리, 배치 N x 단어 순서 T 행렬 받고
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                # 대표 가중치/경사도 행렬 뽑고, 나머지는 다 참조로 연결된 상태...
                params, grads = remove_duplicate(model.params, model.grads)
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 퍼플레서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    # 230쪽 식 5.12, 13 참고
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플레서티 %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            ppl,
                        )
                    )
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel("반복 (x" + str(self.eval_interval) + ")")
        plt.ylabel("퍼플레서티")
        plt.show()


# in_layers 10개(window 5 * 2) 중복 + ns_loss 6개(정답 1 + sample_size 5) 중복 등,
# 같은 가중치를 공유해서 여러 레이어들을 만들고,
# 굳이 그 가중치들을 다 리스트로 모아서 중복을 만들었는데,
# 왜 굳이 이렇게 일부러 중복을 만들고, 여기선 또 그 중에 대표만 뽑아서 얕은 복사로 객체를 만드는지...
# 아래 같은 행렬이 나오면 그걸 빼고, 그에 해당하는 기울기를 대표에게 다 몰아주는데, 이걸 안해서 학습이 안되나보다...
def remove_duplicate(params, grads):
    # 이게 복잡한데, 일단 왼쪽 params와 오른쪽은 다른 객체(메모리 주소 다름) 맞는데,
    # 그 안에 params[0]은 얕은 복사로 왼쪽 오른쪽이 연결된 상태...
    # 아래에서 params.pop 하면, 원래 params 아닌 새로 만든 params에서 그 원소가 빠지는 건 맞는데,
    # 남아있는 params[0]은 원래 params에 들어있던 0번째 배열 그걸 가리키고 있는 상황이다...
    params, grads = params[:], grads[:]  # copy list

    # for문으로만 돌면 중복 pop 시키면 꼬이니까, 중복 나오면 for문 종료하고 밖에 while 문에 의지...
    while True:
        # 중복 제거 for문 시작할 때마다 깃발과 총 길이 다시 설정하고 돌기...
        find_flg = False
        L = len(params)

        # 반복 없이 각 행렬별로 순서쌍 만들기...
        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # params 내 어떤 행렬이 다른 행렬과 메모리 주소가 같을 때...
                # 레이어 만들 때 W_in 하나 만들어서 다 그걸 넣었으면 이런 상황인데...
                if params[i] is params[j]:
                    # 같은 행렬이 나왔다면 분기했다는 의미...역전파는 경사를 더한다..
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
