# KNN-K近邻算法

标签（空格分隔）： 机器学习,KNN

---
算法流程    

> * 计算当前点与样本集合中各点的距离    
> * 按照距离递增排序    
> * 选取与当前点距离最小的k个点 
> * 统计k个点中，各个类别出现的频率     
> * 返回k个点中出现频率最高的类别，作为当前点的预测分类    

代码部分
```python
#coding: utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
def classify0(inputX,dataSet,labels,k):
    #获取样本数量
    #output:4
    dataSetSize = dataSet.shape[0]  
    #把输入向量扩展成矩阵，与样本矩阵做差
    diffMat = tile(inputX, (dataSetSize,1)) - dataSet 
    #差矩阵平方
    sqDiffMat = diffMat**2
    #差矩阵求行和
    sqDistances = sqDiffMat.sum(axis=1) 
    #各行和开方，即欧式距离
    distances = sqDistances**0.5
    #对距离排序，得到下标顺序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    #统计各个label（类别）出现的次数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #按各个label（类别）出现的次数排序，从大到小
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount
```
测试代码
```python
# coding: UTF-8
import kNN

group,labels = kNN.createDataSet()
result = kNN.classify0([0,0],group,labels,3)
print(result)
```




