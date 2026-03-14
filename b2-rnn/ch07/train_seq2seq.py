import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


# 데이터셋 읽기 - 데이터 50,000 x 7 or 5(각각 문자 ID)
(x_train, t_train), (x_test, t_test) = sequence.load_data("addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

# 입력 반전 여부 설정이라...나중에 볼 트릭인가...
is_reverse = True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# 하이퍼파라미터 설정 - 근데 여기 wordvec이나 max_grad 등의 하이퍼파라미터는 어떻게 정하지?
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# 모델
# model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        # x_test[i]를 [[]] 2차원 배열로 뽑아내기...
        question, correct = x_test[[i]], t_test[[i]]
        # 10개 이하면 verbose하게 출력하란 얘기겠지..
        verbose = i < 10
        correct_num += eval_seq2seq(
            model, question, correct, id_to_char, verbose, is_reverse
        )

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print("val acc %.3f%%" % (acc * 100))

# 그래프 그리기
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker="o")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.show()
