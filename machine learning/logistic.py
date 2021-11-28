# -*- coding: utf-8 -*-
from numpy import*
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(float(lineArr[2])))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def plotBestFit(dataArr, labelMat,weights,recognitionLate,name):
    xMat = np.mat(dataArr)
    plt.figure()
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(shape(xMat)[0]):
        if labelMat[i] == 1:
            xcord1.append(xMat[i, 1])
            ycord1.append(xMat[i, 2])
        else:
            xcord2.append(xMat[i, 1])
            ycord2.append(xMat[i, 2])
    plt.scatter(xcord1, ycord1, s=10, c='blue', marker='+', label='x1')
    plt.scatter(xcord2, ycord2, s=10, c='red', marker='^', label='x2')
    min_x = xMat[:, 1].min() - 0.3
    max_x = xMat[:, 1].max() + 0.3
    x = linspace(min_x, max_x, 1000)
    yHat = (-weights[0] - weights[1] * x) / weights[2]
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(mat(x).T, yHat.T, 'r', label='回归线')
    plt.xlabel('$\mathrm{X}_{1}$', fontsize=12)
    plt.ylabel('$\mathrm{X}_{2}$', fontsize=12)
    plt.legend(loc='best', prop={'family': 'SimHei', 'size': 12})
    plt.title('线性逻辑回归分类(' + name + ':' + str(shape(xMat)[0]) + ',识别率:' + str(recognitionLate) + '%)',
              fontproperties='SimHei', fontsize=15)
    plt.grid(True, linestyle="--", color="k")

    plt.show()
def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
def CaculateMse(xArr,yArr,weights):
    dataMatrix=mat(xArr)

    labelMat=mat(yArr).transpose()
    yHat = dataMatrix * weights
    ErroMat = np.zeros((len(set(yArr)), len(set(yArr))))
    for i in range(shape(dataMatrix)[0]):
        if ((labelMat[i] == 1) and (yHat[i] >= 0)):
            ErroMat[0, 0] = ErroMat[0, 0] + 1
        elif (labelMat[i] == 1 and yHat[i] <= 0):
            ErroMat[0, 1] = ErroMat[0, 1] + 1
        elif (labelMat[i] == -1 and yHat[i] >= 0):
            ErroMat[1, 0] = ErroMat[1, 0] + 1
        elif (labelMat[i] == -1 and yHat[i] <= 0):
            ErroMat[1, 1] = ErroMat[1, 1] + 1
    recognitionLate = ((ErroMat[0, 0] + ErroMat[1, 1]) / (
            ErroMat[0, 0] + ErroMat[0, 1] + ErroMat[1, 0] + ErroMat[1, 1])) * 100
    return recognitionLate
def main():
    TrainDataMat, TrainLabelMat = loadDataSet('train_data.text')
    print('TrainDataMat:\n', TrainDataMat)
    print('TrainLabelMat:\n', TrainLabelMat)
    weights = gradAscent(TrainDataMat, TrainLabelMat)
    print('weights:\n', weights.tolist())
    recognitionLate = CaculateMse(TrainDataMat, TrainLabelMat, weights)
    plotBestFit(TrainDataMat, TrainLabelMat,weights,recognitionLate,"训练样本")
    TestDataMat, TestLabelMat = loadDataSet('test_data.text')
    recognitionLate = CaculateMse(TestDataMat, TestLabelMat, weights)
    plotBestFit(TestDataMat, TestLabelMat, weights, recognitionLate, "测试样本")
if __name__ == '__main__':
    main()

