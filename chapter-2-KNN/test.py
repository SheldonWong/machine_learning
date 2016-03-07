# coding: UTF-8
import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from sympy.physics.quantum.circuitplot import pyplot


'''
test1:KNN算法核心部分
'''
def test1():
    group,labels = kNN.createDataSet()
    print(group)
    result = kNN.classify0([0,0],group,labels,3)
    print(result)


'''
test2:画散点图，利用第一列和第二例数据
'''
def test2():
    datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
    print(array(datingLabels))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    plt.show()

'''
先用这种方法测试吧...
'''        
if __name__=="__main__":
    test2()








    