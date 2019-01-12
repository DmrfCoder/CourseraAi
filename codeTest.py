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
    return x - y


def testContourf():
    # 数据数目
    n = 256
    # 定义x, y
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)

    # 生成网格数据
    X, Y = np.meshgrid(x, y)

    # 填充等高线的颜色
    plt.contourf(X, Y, f(X, Y), 4, alpha=0.75, cmap=plt.cm.Spectral)
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
    d_x = np.divide(x, 2)
    print(d_x)
    d_x2 = x / 2
    print(d_x2)

    d_x3 = np.true_divide(x, 2)
    print(d_x3)

    d_x4 = np.floor_divide(x, 2)
    print(d_x4)

    d_x5 = x // 2
    print(d_x5)


def testDropout():
    keep_prob = 0.5
    np.random.seed(1)
    a = np.random.rand(4, 6)
    print(a)
    d1 = np.random.rand(4, 6)

    d1 = d1 < keep_prob

    w = np.random.rand(4, 6)

    a = np.multiply(a, d1)
    print('')
    print(a)
    print('')

    a2 = w * a + 2

    a = a / keep_prob
    a3 = w * a + 2
    m1 = np.mean(a2)
    print(a)
    m2 = np.mean(a3)


def testInitW():
    np.random.seed(1)
    a = np.random.randn(4, 6)
    fangcha1 = a.var()
    print(fangcha1)
    a = a * np.sqrt(1 / 4)
    fangcha2 = a.var()
    print(fangcha2)
    print(0.25 ** 2)


def testSquare():
    print('')
    x = np.arange(5, 10)
    print(x)
    y = np.square(x)
    print(y)


def testLinalgNorm():
    print('')
    x = np.array([[1, 2, 3],
                  [4, 5, 6]])

    # 求的是列的和的最大值，结果应为max(5,7,9)=9
    a1 = np.linalg.norm(x, ord=1, keepdims=False)
    print(a1)

    a2 = np.linalg.norm(x, ord=2, keepdims=False)
    print(a2)

    a3 = np.linalg.norm(x, ord=np.inf, keepdims=False)
    print(a3)


def testCopy():
    print('')
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = np.copy(a)
    c = a
    print(b)
    print(a)
    b[0][0] = 100
    print(a)
    c[0][0] = 101
    print(a)
    a[0][0] = 200
    print(b)
    print(c)


def testNpPad():
    print('')
    a = np.random.randn(3, 2)
    #第0轴（行）的左边填1个1，第0轴的右边填2个2
    b = np.pad(a, ((1, 2), (2, 1)), 'constant', constant_values=((1, 2),(3,4)))
    print(a)
    print(b)

def testMeanAverage():
    print('')
    a = [1,2,3]
    c=np.mean(a)
    d=np.average(a,weights=[0.1,0.2,0.7])
    print(a)
    print(c)
    print(d)



if __name__ == '__main__':
    # testContourf()
    print('Key Interrupt Demo')
    print('Please input some chars:')
    a = input()
