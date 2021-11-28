from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 2
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(np.c_[np.ones(shape(xArr)[0]), xArr])
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0:
        print("badMat")
    ws = xTx.I * (xMat.T * yMat)
    return ws


def CaculateMse(xArr, yArr, ws):
    xMat = mat(np.c_[np.ones(shape(xArr)[0]), xArr])
    yMat = mat(yArr).T
    yHat = xMat * ws
    ErroMat = np.zeros((len(set(yArr)), len(set(yArr))))
    for i in range(shape(xMat)[0]):
        if ((yMat[i] == 1) and (yHat[i] >= 0)):
            ErroMat[0, 0] = ErroMat[0, 0] + 1
        elif (yMat[i] == 1 and yHat[i] <= 0):
            ErroMat[0, 1] = ErroMat[0, 1] + 1
        elif (yMat[i] == -1 and yHat[i] >= 0):
            ErroMat[1, 0] = ErroMat[1, 0] + 1
        elif (yMat[i] == -1 and yHat[i] <= 0):
            ErroMat[1, 1] = ErroMat[1, 1] + 1
    recognitionLate = ((ErroMat[0, 0] + ErroMat[1, 1]) / (
                ErroMat[0, 0] + ErroMat[0, 1] + ErroMat[1, 0] + ErroMat[1, 1])) * 100
    return recognitionLate


def picture(xArr, yArr, ws, recognitionLate, name):
    xMat = np.mat(xArr)
    plt.figure()
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(shape(xMat)[0]):
        if yArr[i] == 1:
            xcord1.append(xMat[i, 0])
            ycord1.append(xMat[i, 1])
        else:
            xcord2.append(xMat[i, 0])
            ycord2.append(xMat[i, 1])
    plt.scatter(xcord1, ycord1, s=10, c='blue', marker='+', label='x1')
    plt.scatter(xcord2, ycord2, s=10, c='red', marker='^', label='x2')
    min_x = xMat[:, 0].min() - 0.3
    max_x = xMat[:, 0].max() + 0.3
    x = linspace(min_x, max_x, 1000)
    yHat = (-ws[0] - ws[1] * x) / ws[2]
    shape(yHat)
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.plot(mat(x).T, yHat.T, 'r', label='回归线')
    plt.xlabel('$\mathrm{X}_{1}$', fontsize=12)
    plt.ylabel('$\mathrm{X}_{2}$', fontsize=12)
    plt.legend(loc='best', prop={'family': 'SimHei', 'size': 12})
    plt.title('线性回归分类(' + name + ':' + str(shape(xMat)[0]) + ',识别率:' + str(recognitionLate) + '%)',
              fontproperties='SimHei', fontsize=15)
    plt.grid(True, linestyle="--", color="k")

    plt.show()


def main():
    xTrainArr, yTrainArr = loadDataSet('train_data.text')
    ws = standRegres(xTrainArr, yTrainArr)
    recognitionLate = CaculateMse(xTrainArr, yTrainArr, ws)
    picture(xTrainArr, yTrainArr, ws, recognitionLate, "训练样本")
    xTestArr, yTestArr = loadDataSet('test_data.text')
    recognitionLate = CaculateMse(xTestArr, yTestArr, ws)
    picture(xTestArr, yTestArr, ws, recognitionLate, "测试样本")


if __name__ == '__main__':
    main()
