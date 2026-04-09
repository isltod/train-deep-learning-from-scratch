import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable, Parameter
from dezero import optimizers
import dezero.functions as F
from dezero.layers import Layer
from dezero.models import MLP, VGG16
from dezero.datasets import Spiral, MNIST
from dezero import DataLoader
import dezero
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import cupy as cp
import numpy as np
import time
from dezero import test_mode
from PIL import Image


model = VGG16(pretrained=True)
# x = np.random.randn(1, 3, 224, 224).astype(np.float32)
# model.plot(x)

url = (
    "https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
)
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
# img.show()
# imshow(np.asarray(img))
# plt.show()

x = VGG16.preprocess(img)
print(type(x), x.shape)
x = x[np.newaxis]
print(type(x), x.shape)

with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

# model.plot(x, to_file="vgg.png")
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
