import numpy as np
import matplotlib.pyplot as plt

def func1():
    a = np.random.randn(4, 3)
    b = np.random.randn(1, 3)
    b.reshape((3, 1))
    c = a * b
    print(c.shape)


def func2():
    a=np.random.randn(4,3,2)
    b=np.random.randn(4,1,2)
    c=a*b

    print(c.shape)


def testScatter():
    c1=[1,2,3]
    c2=[1,2,3]
    plt.scatter(c1,c2,s=30,c='red',marker='o',alpha=0.5,label='C1')
    plt.show()


def testGetnumpyArray():
    c=np.array([1,2,4])
    print(c)
    print(c.shape)


def testSqueeze():
    a=np.random.randn(2,3,1)
    b=np.squeeze(a,axis=1)
    print(a.shape)
    print(b.shape)


def testPower():
    a = np.array([1,2,3])
    print(a)
    c=np.array([1,2,3])
    b=np.power(a,c)
    print(b)


if __name__ == '__main__':
    testPower()
