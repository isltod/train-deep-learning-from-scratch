import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for f in self.transforms:
            img = f(img)
        return img


class Normalize:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean = self.mean
        std = self.std
        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)

        return (array - self.mean) / self.std


class Flatten:
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=int):
        self.dtype = dtype
