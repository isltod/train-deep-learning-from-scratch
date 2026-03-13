import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common import config

config.GPU = True

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

batch_size = 160
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25

# 여기서는 제대로 훈련 데이터와 테스트 데이터를 나눠서 진행
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_test, _, _ = ptb.load_data("test")
vocab_size = len(word_to_id)
# 말뭉치와 정답지
xs = corpus[:-1]
ts = corpus[1:]

model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# test 데이터에 대한 퍼플렉서티
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print("테스트 퍼플렉서티: ", ppl_test)

file_name = model.__class__.__name__ + ".pkl"
file_dir = os.path.dirname(__file__)
file_path = os.path.join(file_dir, file_name)
model.save_params(file_name=file_path)
