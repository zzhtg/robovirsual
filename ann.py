import os
import sys
import random

import numpy as np
import cv2

import armorDetect as ad
import lightDetect as ld
import pefermance as pf

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

def position(leftGroup, rightGroup):
    l_pixel = cv2.boxPoints(leftGroup.raw)
    r_pixel = cv2.boxPoints(rightGroup.raw)
    x = sorted(np.append(l_pixel[0:4, 0], r_pixel[0:4, 0]))
    y = sorted(np.append(l_pixel[0:4, 1], r_pixel[0:4, 1]))
    pos = [i for i in [x[0], y[0], x[7], y[7]]]
    return pos, x, y

def cutDigit(image, leftGroup, rightGroup):
    long_l = leftGroup.rect[2]
    long_r = rightGroup.rect[2]
    x, y = position(leftGroup, rightGroup)[1:3]
    h = (long_l+long_r)/2
    x1, y1, x2, y2 = int((y[0]+y[7])/2 - 1.5*h), int((y[0]+y[7])/2 + 1.5*h), int(
                     (x[0]+x[7])/2 - h*0.75), int((x[0] + x[7])/2 + h*0.75)
    min_digit = 0
    max_right_digit = image.shape[0]
    max_left_digit = image.shape[1]
    x1 = min_digit if x1 < min_digit else x1    # 使用三元运算符改写
    y1 = max_right_digit if y1 > max_right_digit else y1
    x2 = min_digit if x2 < min_digit else x2
    y2 = max_left_digit if y2 > max_left_digit else y2
    bias = -ad.angle_bias - 90.0
    image = rotate(image, bias)
    digit = image[x1: y1, x2: y2]
    return digit


def armor_detect(frame, lightgroup):
    """
    输入：group（可能是灯条的矩形最小边界拟合信息）
    功能：一一对比矩形、找到可能的灯条组合作为装甲
    输出：area（可能是装甲的矩形【长宽、左上角坐标,左右灯条长宽平均值】的列表）
    """
    image = frame.copy()
    armorgroup = []
    num = 0
    lens = len(lightgroup)
    for left in range(lens):
        for right in range(left + 1, lens):
            if(lightgroup[left].rect[0] > lightgroup[right].rect[0]):
                left, right = right, left 
            [x_l, y_l, long_l, short_l] = lightgroup[left].rect
            [x_r, y_r, long_r, short_r] = lightgroup[right].rect
            l_, length_dif = ad.length_dif_det(long_l, long_r)   # 长度差距判断：两灯条的长度差 / 长一点的那个长度 < 36%
            if not l_:
                print("length_dif=", length_dif)
                continue
            w_, width_dif = ad.width_dif_det(short_l, short_r)     # 宽度差距判断：两灯条的宽度差 / 长一点的那个长度 < 68%
            if not w_:
                print("width_dif=", width_dif)
                continue
            a_, armor_aspect = ad.armor_aspect_det(x_l, y_l, x_r, y_r, long_l, long_r, short_l, short_r)  # 横纵比判断：2.7~4.5
            if not a_:
                print("armor_aspect=", armor_aspect)
                continue
            l_pixel = cv2.boxPoints(lightgroup[left].raw)
            r_pixel = cv2.boxPoints(lightgroup[right].raw)
            # 第一个y值最大，第三个y值最小，第二个x值最小，第四个x值最大
            vec_mid, vec_light_l, vec_light_r = ad.ortho_pixel(frame, l_pixel, r_pixel)
            o_, ortho_l_value, ortho_r_value, angle_p, dist_group = ad.ortho_angle(vec_mid, vec_light_l, vec_light_r)
            if not o_:
                print("ortho_l_value=", ortho_l_value,"ortho_r_value=", ortho_r_value, "angle_p=", angle_p)
                continue

            digit = cutDigit(image, lightgroup[left], lightgroup[right])
            # cv2.imshow("digit", digit)

            pos = position(lightgroup[left], lightgroup[right])[0]
            dist = dist_group[0]
            armor = ad.Armor(pos, dist, length_dif, width_dif, armor_aspect,
                            [ortho_l_value, ortho_r_value, angle_p],
                            num, lightgroup[left], lightgroup[right])
            armorgroup.append([armor, digit])
    return armorgroup


col = 18
row = 20

filename = "traindata"
def saveData(directory, image):
    global filename, col, row
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (col, row))
    # image = cv2.equalizeHist(image)

    cv2.imshow("number: {0}".format(directory), image)
    key = cv2.waitKey(0) & 0xff
    if key is ord("s"):
        path = os.path.join(os.getcwd(), filename, str(directory))
    elif ord("8") >= key >= ord("0"):
        path = os.path.join(os.getcwd(), filename, str(key - ord("0")))
    elif key is ord("q"):
        print("exit save data")
        sys.exit(0)
    else:
        print("Not save")
        return False

    if not os.path.exists(path):
        os.makedirs(path)
    count = len(os.listdir(path)) + 1
    if count < size_of_data / digit_num:
        cv2.imwrite(os.path.join(path, str(count)), image)
        print("store the file in {0}.jpg".format(os.path.join(path, str(count))), image.shape)
        return True
    else:
        print("{0} file number = max({1})".format(path, size_of_data / digit_num))
        return False

def readData(directory, count):
    global filename
    path = os.path.join(os.getcwd(), filename, str(directory))
    if os.path.exists(os.path.join(path, str(count))):
        img = cv2.imread(os.path.join(path, str(count)), 0)
        return img
    else:
        print("Error: haven`t {0}".format(os.path.join(path, count)))
        sys.exit(0)

size_of_data = 900
digit_num = 9
ann_dist = "ann_train.xml"

def create(hidden = 80):
    global digit_num, col, row
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([col * row, hidden, digit_num]))
    ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    ann.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
    return ann

def record(sample):
    return np.array([sample], dtype = np.float32)

def traindata():
    global digit_num, size_of_data, filename
    data = []
    for directory in os.listdir(os.path.join(os.getcwd(), filename)):
        for count in os.listdir(os.path.join(os.getcwd(), filename, directory)):
            data.append(record(readData(directory, count).reshape(-1)))
    return data

def labeldata():
    global digit_num, size_of_data, filename
    label = []
    for i in os.listdir(os.path.join(os.getcwd(), filename)):
        for j in os.listdir(os.path.join(os.getcwd(), filename, i)):
            labelarray = []
            for k in range(digit_num):
                if (ord(i) - ord("0")) == k:
                    labelarray.append(1)
                else:
                    labelarray.append(0)
            label.append(record(labelarray))
    return label

def train(ann):
    tdata = traindata()
    label = labeldata()
    count = 0
    for t, l in zip(tdata, label):
        print(os.path.join(str(count), str(len(tdata))))
        count += 1
        # print(t, l)
        # print(t.shape, l.shape)
        # print(type(t), type(l))
        ann.train(t, cv2.ml.ROW_SAMPLE, l)
    if ann_dist in os.listdir(os.getcwd()):
        print("Replace an old {0}".format(ann_dist))
    else:
        print("Now, the directory have {0}".format(ann_dist))
    ann.save(os.path.join(os.getcwd(), ann_dist))
    return ann

def saveDigit(cam = 0):
    global size_of_data, digit_num
    cap = cv2.VideoCapture(cam)
    color = 98
    count = 1
    directory = 1
    if not cap.isOpened():
        print("webcam is not open")
        sys.exit(0)
    else:
        while cap.isOpened():
            _, frame = cap.read()
            if not _:
                print("could not read data from webcam")
                sys.exit(0)
            gray, lightGroup = ld.light_detect(frame, color)
            armorGroup = armor_detect(frame, lightGroup)
            cv2.imshow("frame", frame)
            if armorGroup is not None:
                for armor, digit in armorGroup:
                    if count >= size_of_data / digit_num:
                        count = 0
                        directory += 1
                    if saveData(directory, digit):
                        count += 1

            _k, color = pf.key_detect(cap, color)
            if(_k):
                break
            if count * directory > size_of_data:
                break
    cap.release()
    cv2.destroyAllWindows()

def createData():
    global digit_num, size_of_data
    for directory in range(digit_num):
       for count in range(int(size_of_data / digit_num)):
           if not saveData(directory, np.ones(shape=(100, 100)) * directory * 10):
               continue
    ann = create(100)
    train(ann)

def testAnn():
    global ann_dist
    ann = cv2.ml.ANN_MLP_load(os.path.join(os.getcwd(), ann_dist))
    testData = np.float32([np.ones(shape = (20, 18)).reshape(-1) * 1000])
    print(testData)
    cl, x = ann.predict(testData)
    print(cl, x)

if __name__ == "__main__":
    # saveDigit("/dev/video2")
    # createData()
    ann = create(hidden = 720)
    train(ann)
    testAnn()

