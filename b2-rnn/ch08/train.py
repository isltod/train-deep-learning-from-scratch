import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq
from attention_seq2seq import AttentionSeq2seq


# 데이터는 날짜 형식 변환 데이터 - 텍스트를 문자 단위로 ID로 인코딩, 문제 29차원 x 답 11차원
(x_train, t_train), (x_test, t_test) = sequence.load_data("date.txt")
char_to_id, id_to_char = sequence.get_vocab()

# 입력 반전도 사용하고
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 하이퍼파라미터 설정 - 여전히 이건 어떻게 결정하는지...
vocab_size = len(char_to_id)  # 59가지 문자 - 알파벳, 숫자, 공백 등
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

# 모델 3가지 비교...우선 상속이 빈약해보이는데...어떻게 진행되는지...
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(
            model, question, correct, id_to_char, verbose, is_reverse=True
        )

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print("val acc %.3f%%" % (acc * 100))

model.save_params()

# 그래프
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker="o")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(-0.05, 1.05)
plt.show()
