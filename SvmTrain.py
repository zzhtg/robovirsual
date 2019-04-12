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
    return hogdata

def savetrain(hogdata, endcount = 1500, filename = "F:\\traindata", trainmsg = True): # 保存训练集
    """
    function:   训练识别到的数字图像
    :param digit:原始数字图像
    :param endcount:目录下文件到达这个值就会不再保存并输出错误信息
    :param filename:保存目录
    :param preview:打开预览
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

def svmsave(filenum):
    dataset = readdata(filenum)
    responses = np.repeat(np.arange(1, 9), filenum)[:, np.newaxis]
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, responses)
    svm.save('svm_data.dat')

def predictShow(svm, testsample):
    testsample = np.float32(testsample)
    l = np.array([testsample])
    result = svm.predict(l)[1]
    return result

if __name__ == "__main__":
    # train mode
    # svm = cv2.ml.SVM_load('svm_data.dat')
    svmsave(1500) 
    # testsample = np.load("F:\\traindata\\1\\1.npy")
    # predictShow(svm, testsample)
