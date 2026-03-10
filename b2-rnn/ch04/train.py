import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import config

# GPU 사용 여부...켜먼 화면 나갈 수도...
config.GPU = True

import pickle
from common.trainer import Trainer
from common.optimizer import Adam, AdaGrad
from cbow import CBOW
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# 말뭉치, 단어사전 만들고
corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

# 앞/뒤 단어들, 정답지 만들고
contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# 모델 만들고
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
# optimizer = AdaGrad()
trainer = Trainer(model, optimizer)

# 여기가 학습 + 차트...
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 단어벡터, 사전 피클로 저장...모델에서 학습된 결과가 결국 단어벡터인가?
# 이건 입력가중치고, 출력 가중치가 있는데...서로 같나?
word_vecs = model.word_vecs
# word_vecs1 = model.word_vecs1
if config.GPU:
    word_vecs = to_cpu(word_vecs)
    # word_vecs1 = to_cpu(word_vecs1)
# print(word_vecs[0])
# print(word_vecs1[0])

params = {}
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
pkl_file = os.path.join(os.path.dirname(__file__), "cbow_params.pkl")
# pkl_file = os.path.join(os.path.dirname(__file__), "skip_gram_params.pkl")
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)
