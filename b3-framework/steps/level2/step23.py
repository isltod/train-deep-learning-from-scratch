if "__file__" in globals():
    import os, sys

    # 아래 둘 중 하나를 해서 path에 b3-framework 폴더를 등록해야 작동한다...
    sys.path.append("b3-framework")
    # sys.path.append(os.path.join(os.path.dirname(__file__), "..\.."))

import numpy as np

# from dezero.core_simple import Variable
# __init__.py에 import 된 클래스라서 아래처럼 간단하게 사용할 수 있다..
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
