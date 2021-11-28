from numpy import *
import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        LineArr = line.strip().split('\t')
        dataMat.append([float(LineArr[0]), float(LineArr[1])])
        labelMat.append(float(LineArr[2]))
    return np.array(dataMat), np.array(labelMat)
def train(data_set,labels,iter=15):  #used to train
    lr=0.1#学习率
    data_set=np.mat(data_set)
    n=data_set.shape[0]
    m=data_set.shape[1]
    weights=np.zeros(m)
    bias=0

    for k in range(iter):
        for i in range(n):
            y=data_set[i]*np.mat(weights).T+bias
            weights=weights+lr*(labels[i]-sign(y))*data_set[i]
            bias=bias+lr*(labels[i]-sign(y))
    return weights,bias

def predict(data,weights,bias):    #used to predict
    if(weights is not None and bias is not None):
        return np.sign(data*np.mat(weights).T+bias)
    else:
        return 0
def CaculateMse(yMat,yHat):


    labelMat = mat(yMat).transpose()

    ErroMat = np.zeros((len(set(yMat)), len(set(yMat))))
    for i in range(shape(yMat)[0]):
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
def picture(xArr,yArr,w, bias,recognitionLate,name):
    shape(xArr)
    data=np.mat(xArr)
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
    min_x = xMat[:, 1].min() - 0.3
    max_x = xMat[:, 1].max() + 0.3
    x = linspace(min_x, max_x, 2000)
    yHat = -w[:,0]/w[:,1]*x-bias/w[:,1]

    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(mat(x).T, yHat.T, 'r', label='回归线')
    plt.xlabel('$\mathrm{X}_{1}$', fontsize=12)
    plt.ylabel('$\mathrm{X}_{2}$', fontsize=12)
    plt.legend(loc='best', prop={'family': 'SimHei', 'size': 12})
    plt.title('感知器分类(' + name + ':' + str(shape(xMat)[0]) + ',识别率:' + str(recognitionLate) + '%)',
              fontproperties='SimHei', fontsize=15)
    plt.grid(True, linestyle="--", color="k")
    plt.show()
def main():
    TrainDataArr,TrainLabelArr=loadDataSet('train_data.text')
    weights,bias=train(TrainDataArr,TrainLabelArr)
    print("weights is:", weights)
    print("bias is:", bias)
    yHat = predict(np.mat(TrainDataArr),weights,bias)
    print(yHat)
    recognitionLate =CaculateMse(TrainLabelArr, yHat)
    print("recognitionLate is:",recognitionLate)
    picture(TrainDataArr,TrainLabelArr, weights, bias,recognitionLate,"训练样本")
    TestDataArr, TestLabelArr = loadDataSet('test_data.text')
    yHat = predict(np.mat(TestDataArr),weights,bias)
    recognitionLate = CaculateMse(TestLabelArr, yHat)
    picture(TestDataArr, TestLabelArr, weights, bias, recognitionLate, "测试样本")
if __name__ == '__main__':
    main()
