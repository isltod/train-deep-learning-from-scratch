from step01 import Variable


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


if __name__ == "__main__":
    import numpy as np

    class Square(Function):
        def forward(self, x):
            return x**2

    x = Variable(np.array(10))
    # f = Function()
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
