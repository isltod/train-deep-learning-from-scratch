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

        file_name = os.path.join(os.path.dirname(__file__), file_name)

        params = [p.astype(np.float16) for p in self.params]
        if GPU:
            params = [to_cpu(p) for p in params]

        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    # 피클을 읽어서 params와 grads를 cpu 버전으로 바꾸기...
    def load_params(self, file_name=None):
        if file_name is None:
            file_name = self.__class__.__name__ + ".pkl"

        file_name = os.path.join(os.path.dirname(__file__), file_name)

        if not os.path.exists(file_name):
            raise IOError("No such file: " + file_name)

        with open(file_name, "rb") as f:
            params = pickle.load(f)

        params = [p.astype("f") for p in params]
        if GPU:
            params = [to_gpu(p) for p in params]

        for i, param in enumerate(self.params):
            param[...] = to_gpu(params[i])
