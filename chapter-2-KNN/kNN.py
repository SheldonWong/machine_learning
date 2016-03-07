#coding: utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
'''
@summary: KNN核心算法
@param param: inputX,dataSet,labels,k
@return:  sortedClassCount
'''
def classify0(inputX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]  #样本数量
    diffMat = tile(inputX, (dataSetSize,1)) - dataSet   
    sqDiffMat = diffMat**2  #
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount
'''
@summary: 将文件转换成矩阵
@param param:filename 
@return: returnMat,classLabelVector类别标签向量
'''
def file2matrix(filename):
    fr = open(filename)
    #Return a list of lines read,一行是一个元素
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        #[40920,8.326976,0.953952,largeDose]
        #然后将这个列表赋给returnMat的低0行
        returnMat[index,:] = listFromLine[0:3]
        #最后一个元素是类别标签
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector

        
    
    

    