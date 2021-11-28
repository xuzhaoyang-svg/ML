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
def LDA(dataArr, labelArr):
    """
    Linear Discriminant Analysis
    :param dataArr:
    :param labelArr:
    :return: parameter w
    """
    # 0,1两类数据分开
    data1=dataArr[labelArr==1]
    data2=dataArr[labelArr==-1]
    # 求得两类数据的均值向量
    mean1=data1.mean(axis=0,keepdims=True)
    mean2=data2.mean(axis=0,keepdims=True)



    # 得到两种数据的协方差矩阵
    diff1=data1-mean1
    diff2=data2-mean2
    cov1=np.dot(diff1.T,diff1)
    cov2=np.dot(diff2.T,diff2)

    # 得到“类内散度矩阵”
    sw=cov1+cov2
    # 求得参数w
    swInv=np.linalg.inv(sw)
    w=np.dot(swInv,mean2.T-mean1.T)
    y1=np.mat(data1)*np.mat(w)
    y2=np.mat(data2)*np.mat(w)
    my1=y1.mean()
    my2=y2.mean()
    y0=(np.shape(data1)[0]*my1+np.shape(data2)[0]*my2)/(np.shape(data1)[0]+np.shape(data2)[0])
    return w,y0
def picture(dataArr,labelArr,w,y0,recognitionLate,name):
    xMat = np.mat(dataArr)
    plt.figure()
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(shape(xMat)[0]):
        if labelArr[i] == 1:
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
    yHat = -w[0]*x/w[1]+y0
    shape(yHat)
    # plt.rcParams['font.sans-serif'] = 'SimHei'  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(mat(x).T, yHat.T, 'r', label='回归线')
    plt.xlabel('$\mathrm{X}_{1}$', fontsize=12)
    plt.ylabel('$\mathrm{X}_{2}$', fontsize=12)
    plt.legend(loc='best', prop={'family': 'SimHei', 'size': 12})
    plt.title('线性判别分类(' + name + ':' + str(shape(xMat)[0]) + ',识别率:' + str(recognitionLate) + '%)',
              fontproperties='SimHei', fontsize=15)
    plt.grid(True, linestyle="--", color="k")

    plt.show()
def CaculateMse(xArr,yArr,ws,y0):
    xMat = mat(xArr)
    labelMat=mat(yArr).transpose()
    yHat = -xMat*ws-y0

    ErroMat = np.zeros((len(set(yArr)), len(set(yArr))))
    for i in range(shape(xMat)[0]):
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
    TrainDataArr, TrainLabelArr=loadDataSet('train_data.text')
    w,y0= LDA(TrainDataArr, TrainLabelArr)
    print("w=",w.tolist())
    print("y0=", y0)
    recognitionLate=CaculateMse(TrainDataArr,TrainLabelArr,w,y0)
    picture(TrainDataArr, TrainLabelArr,w,y0,recognitionLate,"训练样本")
    TestDataArr, TestLabelArr = loadDataSet('test_data.text')
    recognitionLate = CaculateMse(TestDataArr, TestLabelArr, w, y0)
    picture(TestDataArr, TestLabelArr, w, y0, recognitionLate, "测试样本")
if __name__ == '__main__':
    main()

