import sys, os

sys.path.append("..")
import pickle
from common.np import *
from common.util import to_cpu, to_gpu


class BaseModel:
    def __init__(self):
        self.params, self.grads = None, None

    # 상속 받은 후에 forward와 backward는 오버로드 하지 않으면 오류 내는 코드인가?
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    # params와 grads를 cpu 버전으로 바꿔서 피클로 저장
    def save_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        params = [to_cpu(p) for p in self.params]
        grads = [to_cpu(g) for g in self.grads]

        with open(file_name, "wb") as f:
            pickle.dump((params, grads), f)

    # 피클을 읽어서 params와 grads를 cpu 버전으로 바꾸기...
    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        if not os.path.exists(file_name):
            raise IOError("No such file: " + file_name)

        with open(file_name, "rb") as f:
            params, grads = pickle.dump(f)

        for i, param in enumerate(self.params):
            param[...] = to_gpu(params[i])

        for i, grad in enumerate(self.grads):
            grad[...] = to_gpu(grads[i])
