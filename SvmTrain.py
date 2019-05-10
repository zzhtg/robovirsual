import cv2
import os
import time
import numpy as np
import armorDetect as ad


def rotate(image, angle, center=None, scale=1.0): #1
    '''
    function:   根据输入角度将图像矫正
    :param image: 输入图像
    :param angle: 矫正角度
    :param center: 中心坐标
    :param scale: 放大倍率
    :return: 旋转后的图像
    '''
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4
    M = cv2.getRotationMatrix2D(center, angle, scale) #5
    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated

bin_n = 16
def hog(img):
    """
    function:   根据输入图像输出特征描述符
    :param img:输入图像
    :return:特征
    """
    # 获取图像的hog特征 hog：梯度直方图
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # sobel算子
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)     # 计算梯度的幅度和相位
    bins = np.int32(bin_n*ang/(2*np.pi))   # quantizing binvalues in (0...16)
    # 将范围映射到0-15.待会统计直方图的时候就有16组，组数越多，精度越高
    bin_cel_ls = bins[:10,:9], bins[10:,:9], bins[:10,9:], bins[10:,9:]
    mag_cel_ls = mag[:10,:9], mag[10:,:9], mag[:10,9:], mag[10:,9:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in list(zip(bin_cel_ls, mag_cel_ls))]
    # bincount 为统计频数函数，给出最小值到最大值每一个值出现的频数
    # arg1：要统计的数据，arg2：每个数据的权值，arg3：最小值
    hist = np.hstack(hists) # hist is a 64 bit vector
    return hist

def image2hog(digit, preview = False):
    digit_b, digit_g, digit_r = cv2.split(digit)
    _, gg = cv2.threshold(cv2.equalizeHist(digit_g), 160, 255, cv2.THRESH_BINARY)
    bias = -ad.angle_bias - 90.0
    gg1 = rotate(gg, bias)
    traininput = cv2.resize(gg1, (18, 20))
    if (preview):
        cv2.imshow("preview_digits", cv2.resize(gg1, (400, 400)))
        cv2.imshow("preview_digits_rotated", cv2.resize(gg1, (400, 400)))
        cv2.imshow("preview_traininput", cv2.resize(traininput, (400, 400)))
    hogdata = hog(traininput)
    return hogdata, traininput

def saveimage(img, endcount = 500, filename = "F:\\trainimg", trainmsg = True): # 保存训练集
    """
    function:   训练识别到的数字图像
    :param digit:原始数字图像
    :param endcount:目录下文件到达这个值就会不再保存并输出错误信息
    :param filename:保存目录
    :param trainmsg:打开保存信息输出预览，显示保存到第几个文件
    :return:
    """
    # 要统计的文件夹
    if (os.path.exists(filename) == False):
        print("该目录不存在，创建目录为%s"%filename)
        os.makedirs(filename)
        time.sleep(2)
    filecount = len([name for name in os.listdir(filename) if os.path.isfile(os.path.join(filename, name))])
    if(filecount >= endcount):
        print("The number of data file has over the limit: %d files"%(endcount))
        return
    trainname = (filename + "\\" + str(filecount) + ".jpg")
    cv2.imwrite(trainname, img)
    if(trainmsg):
        print(trainname + " have saved")

def savetrain(hogdata, endcount = 500, filename = "F:\\traindata", trainmsg = True): # 保存训练集
    """
    function:   训练识别到的数字图像
    :param digit:原始数字图像
    :param endcount:目录下文件到达这个值就会不再保存并输出错误信息
    :param filename:保存目录
    :param trainmsg:打开保存信息输出预览，显示保存到第几个文件
    :return:
    """
    # 要统计的文件夹
    if (os.path.exists(filename) == False):
        print("该目录不存在，创建目录为%s"%filename)
        os.makedirs(filename)
        time.sleep(2)
    filecount = len([name for name in os.listdir(filename) if os.path.isfile(os.path.join(filename, name))])
    if(filecount >= endcount):
        print("The number of data file has over the limit: %d files"%(endcount))
        return
    trainname = (filename + "\\" + str(filecount) + ".npy")
    np.save(trainname, hogdata)
    if(trainmsg):
        print(trainname + " have saved")

def readdata(filenum, file = "F:\\traindata"):
    count = 0
    traindata = np.zeros((filenum * 8, 64), np.float32)
    group = [(file + "\\" + str(i) + "\\" ) for i in range(1, 9)]
    for thisnum in range(0, 8):
        for samplenum in range(0, filenum):
            traindata[count, :] = np.load(group[thisnum] + str(samplenum) + ".npy")
            count = count + 1
    return traindata
def readimg(filenum, file = "F:\\trainimg"):
    count = 0
    traindata = np.zeros((filenum * 9, 18 * 20), np.float32)
    group = [(file + "\\" + str(i) + "\\") for i in range(0, 9)]
    for thisnum in range(0, 9):
        for samplenum in range(0, filenum):
            img = cv2.imread(group[thisnum] + str(samplenum) + ".jpg", 0)
            img = img.reshape((1, 18 * 20))
            traindata[count, :] = img
            count = count + 1
    return traindata
def svmsave_img(filenum):
    dataset = readimg(filenum)
    responses = np.repeat(np.arange(0, 9), filenum)[:, np.newaxis]
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
    return svm, dataset, responses

def svmsave(filenum):
    dataset = readdata(filenum)
    responses = np.repeat(np.arange(0, 9), filenum)[:, np.newaxis]
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setCoef0(0)
    # svm.setCoef0(0.0)
    # svm.setDegree(3)
    # criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    # svm.setTermCriteria(criteria)
    # svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setNu(0.5)
    # svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    # svm.setC(0.01)  # From paper, soft classifier
    # svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task

    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')
    return svm, dataset, responses

def predictShow(svm, testsample):
    testsample = np.float32(testsample)
    testsample = testsample.reshape((1, 18 * 20))
    # l = np.array([testsample])
    result = svm.predict(testsample)[1]
    return result

if __name__ == "__main__":
    # train mode
    svm, dataset, responses = svmsave_img(500)
    # testsample = cv2.imread("F:\\trainimg\\1\\0.jpg", 0)
    result = svm.predict(dataset)[1]
    mask = result == responses
    correct = np.count_nonzero(mask)
    print(correct * 100.0 / result.size)