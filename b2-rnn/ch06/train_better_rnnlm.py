import pickle
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common import config

config.GPU = True

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
from dataset import ptb
from better_rnnlm import BetterRnnlm

# 하이퍼 파라미터인데...이건 왜 이렇게 주는지 어떻게 결정하는건가...
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
# 원래 40번 돌릴려고 했는데...자꾸 죽으니 10번씩 나눠서 돌려보자...27
# max_epoch = 40
max_epoch = 10
max_grad = 0.25
dropout = 0.5

# 예의 그 데이터 준비...다만 train, validate, test 나눠서...
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_val, _, _ = ptb.load_data("val")
corpus_test, _, _ = ptb.load_data("test")

# GPU면 cupy 배열로...
if config.GPU:
    corpus = to_gpu(corpus)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

# 문제와 정답
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 열 몇 번 에포크에서 자꾸 죽어서 열 번 돌아가고 저장하고, 다시 읽어들여서 시작하도록 변경해보자...
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
file_dir = os.path.dirname(__file__)
file_name = model.__class__.__name__ + ".pkl"
file_path = os.path.join(file_dir, file_name)
lr_file = os.path.join(file_dir, "lr.pkl")
ppl_file = os.path.join(file_dir, "ppl.pkl")
# 맨 처음 이외에는 아래를 실행
model.load_params(file_path)
if os.path.exists(lr_file):
    with open(lr_file, "rb") as f:
        lr = pickle.load(f)
if os.path.exists(ppl_file):
    with open(ppl_file, "rb") as f:
        best_ppl = pickle.load(f)

optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float("inf")
for epoch in range(max_epoch):
    # 한 번만 학습하고...
    trainer.fit(
        xs,
        ts,
        max_epoch=1,
        batch_size=batch_size,
        time_size=time_size,
        max_grad=max_grad,
    )
    # 상태를 reset 시키는데...이래도 되나? epoch는 데이터 소진이니까 다시 시작하는게 맞나?
    # 그럼 거꾸로 이전에 이렇게 안 했을때가 문젠가? 그냥 이런 정도는 대충하는건가?
    model.reset_state()
    # validate 데이터로 퍼플렉서티를 계산...
    ppl = eval_perplexity(model, corpus_val)
    print("검증 퍼플렉서티: ", ppl)

    if ppl < best_ppl:
        best_ppl = ppl
        with open(ppl_file, "wb") as f:
            pickle.dump(best_ppl, f)
        model.save_params(file_name=file_path)
    else:
        lr /= 4.0
        optimizer.lr = lr
        # 10 에포크씩 나눠 돌리고, 다시 들어오면 이전 lr 읽어야 하니까 바뀔 때마다 저장...
        with open(lr_file, "wb") as f:
            pickle.dump(lr, f)

    model.reset_state()
    print("=" * 50)


model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print("테스트 퍼플렉서티: ", ppl_test)
