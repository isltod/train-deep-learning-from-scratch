import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파라미터 설정
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN 은닉 상태 벡터의 원소 수
time_size = 5  # RNN을 펼치는 크기 - Truncated BPTT가 한 번에 펼치는 시간(단어 수) 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data("train")
# 전체 중 1001개만, 나중 계산 편하게... - 현재 상태로는 더 큰 말뭉치는 안된다고...왜지?
corpus_size = 1001
corpus = corpus[:corpus_size]
# 모든 데이터 사용할 때는 word_to_id 등의 len을 보면 됐지만,
# 지금은 작게 잘라서 사용하므로 새로 단어사전 만들고 길이 보고 ID들 다 고치고...복잡하니까 제일 큰 ID로...
vocab_size = int(max(corpus) + 1)
# 다음 단어 예측 모델이라서 처음부터 -1번째 까지 훈련 데이터의 정답지는 2번째부터 마지막 단어까지...
xs = corpus[:-1]
ts = corpus[1:]

# 모델 만들고 결과
# 단어 사전 크기, 단어 표현 차원, 히든 크기..
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 훈련 데이터(처음부터 -1까지 문장), 정답지(2에서 마지막까지 문장), 100 에포크 돌기, 10개씩 끊어 처리, RNN 펼치기 5개
trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot(ylim=(0, 500))
