import numpy as np
import matplotlib.pyplot as plt


def func1():
    a = np.random.randn(4, 3)
    b = np.random.randn(1, 3)
    b.reshape((3, 1))
    c = a * b
    print(c.shape)


def func2():
    a = np.random.randn(4, 3, 2)
    b = np.random.randn(4, 1, 2)
    c = a * b

    print(c.shape)


def testScatter():
    c1 = [1, 2, 3]
    c2 = [1, 2, 3]
    plt.scatter(c1, c2, s=30, c='red', marker='o', alpha=0.5, label='C1')
    plt.show()


def testGetnumpyArray():
    c = np.array([1, 2, 4])
    print(c)
    print(c.shape)


def testSqueeze():
    a = np.random.randn(2, 3, 1)
    b = np.squeeze(a, axis=1)
    print(a.shape)
    print(b.shape)


def testPower():
    a = np.array([1, 2, 3])
    print(a)
    c = np.array([1, 2, 3])
    b = np.power(a, c)
    print(b)


def testMeshgrid():
    a = [1, 2, 3]
    b = [4, 5, 6]
    print(a)
    print(b)
    [x, y] = np.meshgrid(a, b)
    print(x)
    print(y)
    plt.figure()
    plt.scatter(x, y)
    plt.show()


def testNumpy_r_c():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    c = np.r_[a, b]
    d = np.c_[a, b]
    print(a)
    print(b)
    print(c)
    print(d)


def test_ravel_flatten():
    x = np.array([[1, 2], [3, 4]])
    print(x)
    print()

    a = x.flatten()
    print(x)
    print(a)
    x[0][0] = 10
    x[0][1] = 20
    print(a)

    a[0] = -1
    print(x)
    print()

    b = x.ravel()
    print(x)
    print(b)
    x[0][0] = 30
    x[0][1] = 40
    print(b)
    b[0] = -1
    print(x)


# 定义等高线高度函数
def f(x, y):
    return x-y


def testContourf():
    # 数据数目
    n = 256
    # 定义x, y
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)

    # 生成网格数据
    X, Y = np.meshgrid(x, y)

    # 填充等高线的颜色
    plt.contourf(X, Y, f(X, Y), 4,alpha=0.75, cmap=plt.cm.Spectral)
    # 绘制等高线
    C = plt.contour(X, Y, f(X, Y), 4, colors='black', linewidth=0.5)
    # 绘制等高线数据
    plt.clabel(C, inline=True, fontsize=10)

    # 去除坐标轴
    plt.xticks(())
    plt.yticks(())
    plt.show()



def testDivide():
    print()
    x = np.array([[1, 3], [5, 7]])
    print(x)
    d_x=np.divide(x,2)
    print(d_x)
    d_x2=x/2
    print(d_x2)

    d_x3=np.true_divide(x,2)
    print(d_x3)

    d_x4=np.floor_divide(x,2)
    print(d_x4)

    d_x5=x//2
    print(d_x5)




if __name__ == '__main__':
    testContourf()
