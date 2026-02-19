import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01 * x**2 + 0.1 * x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    # d*x + b = f(x) 이므로 y 절편 b를 구한다...
    b = f(x) - d * x
    # 접점에서 만나는 접선의 함수식을 람다식으로 반환
    return lambda t: d * t + b


if __name__ == "__main__":
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

    print(numerical_diff(function_1, 5))
    print(numerical_diff(function_1, 10))

    # x = 5에서 만나는 접선의 함수식을 받아서...
    tf = tangent_line(function_1, 5)
    y2 = tf(x)
    # 원래 함수
    plt.plot(x, y)
    # 접선
    plt.plot(x, y2)
    # 접점
    plt.plot(5, tf(5), "o")
    # 수직/수평 점선 (접점 위치)
    plt.axvline(x=5, color="gray", linestyle="dashed", linewidth=1)
    plt.axhline(y=tf(5), color="gray", linestyle="dashed", linewidth=1)
    plt.show()

    tf = tangent_line(function_1, 10)
    y2 = tf(x)
    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(10, tf(10), "o")
    plt.axvline(x=10, color="gray", linestyle="dashed", linewidth=1)
    plt.axhline(y=tf(10), color="gray", linestyle="dashed", linewidth=1)
    plt.show()
