import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.np import *  # import numpy as
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    # h는 은닉층 안예 계산되어 있는 값, W는 은닉층 가중치,
    # 거기서 h와 같은 행을 뽑는데, 그 인덱스들이 idx, 뽑은 결과가 target_W
    def forward(self, h, idx):
        # embed 층에 idx를 주고 forward하면 해당 인덱스의 열을 벡터로 추출,
        # idx는 미니배치로 배열 형태
        target_W = self.embed.forward(idx)
        # 이게 dot, 미니배치 고려해서 원소별 곱을 행 단위로 합, 그림 4-14
        out = np.sum(target_W * h, axis=1)
        # h와 target_W만 저장?
        self.cache = (h, target_W)
        return out

    # 역전파는 내가 생각해보라고? 간단하지 않은거 같은데...
    def backward(self, dout):
        # 일단 저장해놓은 은닉층 원래 값, 거기에 해당되는 가중치 원래 값
        h, target_W = self.cache
        # 위에서 내려온 미분값을 열벡터로 만든다?
        dout = dout.reshape(dout.shape[0], 1)

        # target_W 방향 미분은 반대쪽 h를 곱한다...이전에 하던대로...
        dtarget_W = dout * h
        # embed의 backward는 전달한 미분값을 계산에 참여한 행에만 누적해 더하고,
        # 나머지는 0으로 처리하는게 다인데...
        # 그럼 W 방향 역전파는 위에서 내려온 미분 dout에 반대쪽 h를 곱한 값들이
        # 원래 W 모양의 행렬에다 계산에 참여한 행들에만 들어가 있는 모양인데...
        # 그게 여기가 아니고 embed의 dW에 있는데...나중에 optimizer에서 이걸 쓰나?
        self.embed.backward(dtarget_W)
        # h 방향 역전파는 반대방향을 곱해서 보낸다..근데 target_W는 T가 아니네?
        dh = dout * target_W
        # 그리고 h 방향 역전파만 반환? 이거만 중요한가?
        return dh


# 단어별 확률에 따라 오답지 단어를 샘플링해주는 클래스
class UnigramSampler:
    # 말뭉치, 희소 단어 확률 보정 지수, 오답 단어 샘플 갯수 받아서 단어별 확률 배열 만들기
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        # 각 단어가 몇 개 있는지 세서 딕셔너리로 만들기
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        # 중복 없이 센 단어 수
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        # 단어 총 수(어휘 수 말고)로 단어 출현 확률을 구하는데...
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            # counts가 사전이라고 했는데...리스트처럼 슬라이싱이 되나?
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    # 정답 단어 배치 받아서, 각 단어별로 sample_size만큼 오답을 샘플링해서 반환
    # 총 오답 수는 배치 x sample_size
    def get_negative_sample(self, target):
        # 정답지의 첫 번째 차원이 배치
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                # 정답은 뽑지 않도록 확률 0으로 만들고, 다시 전체 합이 1이 되도록 수정...
                p[target_idx] = 0
                p /= p.sum()
                # 어휘 수 한도 내에서, 반복 없이 확률 p에 따라 sample_size 만큼 숫자 뽑기 - 인덱스 배열
                negative_sample[i] = np.random.choice(
                    self.vocab_size, size=self.sample_size, replace=False, p=p
                )
        else:
            # cupy에서는 반복도 허용하고 정답 확률을 0으로 만들지도 않는다...속도를 우선한다고...
            # 근데 이러면 정답지가 오답에 포함될 수 도 있는 문제가 있다...
            negative_sample = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.sample_size),
                replace=True,
                p=self.word_p,
            )

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        # loss는 손실 계산 층, embed는 embedding해서 dot 곱하는 층, 모두 정답 1 + 오답 sample_size 만큼 만들기
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        # 오답지 샘플, 위 init에서 만든 sampler로 corpus에서 power 지수로 sample_size 만큼 뽑는다...
        negative_sample = self.sampler.get_negative_sample(target)

        # 첫 번째 층이 정답 층, 정답으로 첫 번째 forward 먼저 계산
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 두 번째부터 마지막까지 오답층, 일단 오답 라벨은 0으로...
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            # 이번에 처리할 오답지는 모든 배치에서 i번째 오답(모든 행에서 i 열)
            negative_target = negative_sample[:, i]
            # 그걸 배치 처리로 넣어서 임베딩 + 닷 곱 -> 손실 계산까지...
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            # 역전파는 손실 -> embed_dot으로 전달되는데, sigmoid with loss는 y - t,
            dscore = l0.backward(dout)
            # 하나의 은닉층 h를 sample_size + 1만큼 반복해서 곱해서 나눠줬으므로,
            # 반대는 각 손실을 누적...zip은 0-0, 1-1, 2-2 등으로 묶어주는 함수
            dh += l1.backward(dscore)

        return dh
