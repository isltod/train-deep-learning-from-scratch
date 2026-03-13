import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common import config

config.GPU = True

from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = BetterRnnlmGen()
param_file = os.getcwd() + "\\b2-rnn\\ch06\\BetterRnnlm.pkl"
model.load_params(param_file)

start_word = "you"
start_id = word_to_id[start_word]
skip_words = ["N", "<unk>", "$"]
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = " ".join([id_to_word[i] for i in word_ids])
txt = txt.replace(" <eos>", ".\n")
print(txt)

model.reset_state()

start_words = "the meaning of life is"
start_ids = [word_to_id[w] for w in start_words.split(" ")]

# is 빼고 앞에서부터 순서대로 순전파 시키기...
for x in start_ids[:-1]:
    x = np.array(x).reshape((1, 1))
    model.predict(x)

# 마지막 is 넣고 generate...
word_ids = model.generate(start_ids[-1], skip_ids)
# 생성된 문장 리스트에 the meaning of life 까지 붙이고
word_ids = start_ids[:-1] + word_ids
txt = " ".join([id_to_word[i] for i in word_ids])
txt = txt.replace(" <eos>", ".\n")
print("-" * 50)
print(txt)
