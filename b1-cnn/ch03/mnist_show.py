import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.mnist import load_mnist
from PIL import Image
import numpy as np


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    # WSL2에서 이거 작동하려면 sudo apt install imagemagick -y
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

label = t_train[0]
print(label)

img = x_train[0]
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
#
