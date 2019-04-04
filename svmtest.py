import cv2
import numpy as np
from matplotlib import pyplot as plt
def deskew(img):
    '''
    :param img: 原始图像信息
    :return: 返回值：矫正之后的图像 '''
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags) # 仿射变换 参数：原始图像，
    return img

def hog(img):
    # 获取图像的hog特征 hog：梯度直方图
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # sobel算子
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)     # 计算梯度的幅度和相位
    bins = np.int32(bin_n*ang/(2*np.pi))   # quantizing binvalues in (0...16)
    # 将范围映射到0-15.待会统计直方图的时候就有16组，组数越多，精度越高
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in list(zip(bin_cells, mag_cells))]
    # bincount 为统计频数函数，给出最小值到最大值每一个值出现的频数
    # arg1：要统计的数据，arg2：每个数据的权值，arg3：最小值
    hist = np.hstack(hists) # hist is a 64 bit vector
    return hist

if __name__ == "__main__":
    img = cv2.imread("digits.png", 0)
    global SZ
    SZ = 20  # 一个数字的长和宽都是20
    bin_n = 16 # 直方图分组个数
    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
    # 原始图像像素 2000 x 1000
    # 一个数字像素 20 x 20
    # 总共数字个数 100 x 50， 即100列 50行， 左50列训练，右50列测试， 一个数字5行
    # First half is trainData, remaining is testData
    train_cells = [i[:50] for i in cells]
    test_cells = [i[50:] for i in cells]
    ###### Now training ########################
    deskewed = [list(map(deskew, row)) for row in train_cells]    # deskew 对row的每一个元素做矫正
    hogdata = [list(map(hog, row)) for row in deskewed]           # hog 对row的每一个元素做处理
    trainData = np.float32(hogdata).reshape(-1, 64)               # 转换成float类型
    responses = np.repeat(np.arange(10), 250)[:, np.newaxis]      # 标准答案
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    # svm.save('svm_data.dat')
    # ####### Check Accuracy ########################
    deskewed = [list(map(deskew, row)) for row in test_cells]
    hogdata = [list(map(hog, row)) for row in deskewed]
    testData = np.float32(hogdata).reshape(-1, bin_n * 4)
    l = testData[0:1]
    ll = testData[0]
    # print(np.shape(l))
    # print(np.shape(l[0]))
    # print(np.shape(ll[0]))
    result = svm.predict(np.array([ll]))[1]
    # mask = (result == responses)
    # correct = np.count_nonzero(mask)
    print(result)