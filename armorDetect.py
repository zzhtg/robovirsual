# coding=utf-8
import cv2
import numpy as np
import math
import pefermance as pf
import SvmTrain as st
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def length_dif_det(l_left, l_right):
    """
    输入：l_left（左灯条长度）、l_right（右灯条长度）
    功能：检测当前灯条组合是否符合长度差距条件
    输出：True 或者 False 以及 长度宽度比 """
    length_dif = abs(l_left-l_right) / max(l_left, l_right)
    return length_dif <= 1.0, length_dif


def width_dif_det(w_left, w_right):
    """
    输入：w_left（左灯条宽度）w_right（右灯条宽度）
    功能：检测当前灯条组合是否符合宽度差距条件
    输出：True 或者 False 以及 宽度差距比
    """
    w_left, w_right = [i+1 for i in [w_left, w_right]]
    width_dif = abs(w_left-w_right) / max(w_left, w_right)
    return width_dif <= 1.0, width_dif


def armor_aspect_det(x_l, y_l, x_r, y_r, l_l, l_r, w_l, w_r):
    """
    输入：最小矩形拟合信息（x、y、w、h）L（左灯条）（x、y、w、h、）R（右灯条）
    功能：检测当前灯条组合是否符合横纵比条件
    输出：True 或者 False 以及 装甲横纵比
    """
    armor_aspect = math.sqrt((y_r-y_l)**2 + (x_r-x_l)**2) / max(l_l, l_r, w_l, w_r)
    return 4 >= armor_aspect >= 0.9, armor_aspect


def light_dist(l_l, l_r):
    """
    输入：左右灯条的像素长度
    功能：计算摄像头到灯条的距离
    输出：比较近的那个灯条的距离
    """
    l_length_distance = (547.27 * 5.5) / (1 + l_l)
    r_length_distance = (547.27 * 5.5) / (1 + l_r)
    return min(l_length_distance, r_length_distance)


angle_bias = 0
debug_mode = False


def ortho_pixel(frame, l_pixel, r_pixel):
    """
    输入：左右灯条的四个点坐标
    功能：找到左右灯条和中心矢量的对应两个坐标
    输出：中心线、左灯条、右灯条两个坐标
    """
    global debug_mode
    lxmid = np.average(l_pixel[0:4, 0])
    lymid = np.average(l_pixel[0:4, 1])
    rxmid = np.average(r_pixel[0:4, 0])
    rymid = np.average(r_pixel[0:4, 1])

    l_l = squareform(pdist(l_pixel, metric="euclidean"))  # 将distA数组变成一个矩阵
    rr = squareform(pdist(r_pixel, metric="euclidean"))

    ano_l = np.argsort(l_l)     # 对于左灯条的第0个顶点而言，找出能够组成短边的另一个点的序号
    ano_r = np.argsort(rr)      # 对于左灯条的第0个顶点而言，找出能够组成短边的另一个点的序号
    directlxmid, directlymid = (l_pixel[0, :] + l_pixel[ano_l[0, 1], :]) / 2
    directrxmid, directrymid = (r_pixel[0, :] + r_pixel[ano_r[0, 1], :]) / 2

    vec_mid = [[lxmid, lymid],  [rxmid, rymid]]                 # 中心连接矢量
    vec_light_l = [[lxmid, lymid], [directlxmid, directlymid]]  # 灯条1的方向矢量
    vec_light_r = [[rxmid, rymid], [directrxmid, directrymid]]  # 灯条2的方向矢量

    if debug_mode:        # debug 划线及输出
        cv2.line(frame, tuple(vec_mid[0]), tuple(vec_mid[1]), (255, 0, 0), 5)
        cv2.line(frame, tuple(vec_light_l[0]), tuple(vec_light_l[1]), (255, 0, 0), 5)
        cv2.line(frame, tuple(vec_light_r[0]), tuple(vec_light_r[1]), (255, 0, 0), 5)
    return vec_mid, vec_light_l, vec_light_r


def ortho_angle(vec_mid, vec_light_l, vec_light_r):
    """
    输入：获取左右灯条和中心线的两个坐标
    功能：计算两个灯条的中心连接矢量和两个灯条方向矢量的正交性
    输出：True or False , 左右灯条方向矢量与中心连接矢量的夹角
    """
    global angle_bias, debug_mode

    vec_mid = [vec_mid[0][i] - vec_mid[1][i] for i in range(2)]
    vec_light_l = [vec_light_l[0][i] - vec_light_l[1][i] for i in range(2)]
    vec_light_r = [vec_light_r[0][i] - vec_light_r[1][i] for i in range(2)]

    abs_c = math.sqrt(vec_mid[0]**2 + vec_mid[1]**2)
    abs_l = math.sqrt(vec_light_l[0]**2 + vec_light_l[1]**2)
    abs_r = math.sqrt(vec_light_r[0]**2 + vec_light_r[1]**2)

    inl = (vec_mid[0] * vec_light_l[0] + vec_mid[1] * vec_light_l[1])   # 内积
    inr = (vec_mid[0] * vec_light_r[0] + vec_mid[1] * vec_light_r[1])
    inp = (vec_light_l[0]*vec_light_r[0] + vec_light_l[1] * vec_light_r[1])
    angle_l = inl / (abs_c * abs_l)      # 左向量与中心向量的夹角
    angle_r = inr / (abs_c * abs_r)      # 右向量与中心向量的夹角
    angle_p = inp / (abs_l * abs_r)      # 左右向量夹角
    angle_bias = (math.atan2(vec_mid[0], vec_mid[1]) / math.pi * 180.0)
    return_flag = (abs(angle_l) < 0.3 and
                   abs(angle_r) < 0.3 and
                   abs(angle_p) > 0.9)
    if return_flag and debug_mode:
        print("angle_l = ", angle_l, "angle_r = ", angle_r, "midAngle = ", (angle_l + angle_r) / 2)
    # 范围 60~120度， 两个灯条都满足
    return return_flag, angle_l, angle_r, angle_p


def svm_digit_detect(target_num, detect_num):
    return target_num == detect_num, detect_num


def armor_detect(svm, frame, group, target_num, train_mode=False, file="F:\\traindata\\"):
    """
    输入：group（可能是灯条的矩形最小边界拟合信息）
    功能：一一对比矩形、找到可能的灯条组合作为装甲
    输出：area（可能是装甲的矩形【长宽、左上角坐标,左右灯条长宽平均值】的列表）
    """
    image = frame.copy()
    area = []
    lens = len(group)
    for left in range(lens):
        for right in range(left + 1, lens):
            if group[left][0][0] > group[right][0][0]:
                left, right = right, left
            x_l, y_l, w_l, l_l = [j for i in group[left][0:2] for j in i]
            x_r, y_r, w_r, l_r = [j for i in group[right][0:2] for j in i]
            if w_l > l_l:
                w_l, l_l = l_l, w_l
            if w_r > l_r:
                w_r, l_r = l_r, w_r

            l_, length_dif = length_dif_det(l_l, l_r)   # 长度差距判断：两灯条的长度差 / 长一点的那个长度 < 36%
            if not l_:
                print("length_dif=", length_dif)
                continue
            w_, width_dif = width_dif_det(w_l, w_r)     # 宽度差距判断：两灯条的宽度差 / 长一点的那个长度 < 68%
            if not w_:
                print("width_dif=", width_dif)
                continue
            a_, armor_aspect = armor_aspect_det(x_l, y_l, x_r, y_r, l_l, l_r, w_l, w_r)  # 横纵比判断：2.7~4.5
            if not a_:
                print("armor_aspect=", armor_aspect)
                continue
            l_pixel = cv2.boxPoints(group[left])
            r_pixel = cv2.boxPoints(group[right])
            # 第一个y值最大，第三个y值最小，第二个x值最小，第四个x值最大
            vec_mid, vec_light_l, vec_light_r = ortho_pixel(frame, l_pixel, r_pixel)
            o_, ortho_l_value, ortho_r_value, angle_p = ortho_angle(vec_mid, vec_light_l, vec_light_r)  # 垂直判断：< 0.9
            if not o_:
                print("ortho_l_value=", ortho_l_value,"ortho_r_value=", ortho_r_value, "angle_p=", angle_p)
                continue
            x = sorted(np.append(l_pixel[0:4, 0], r_pixel[0:4, 0]))
            y = sorted(np.append(l_pixel[0:4, 1], r_pixel[0:4, 1]))
            armor = [i for i in [x[0], y[0], x[7], y[7]]]

            # digit detection
            h = (l_l+l_r)/2
            x1, y1, x2, y2 = int((y[0] + y[7]) / 2 - h), int((y[0] + y[7]) / 2 + h), int(
                                (x[0] + x[7]) / 2 - h * 0.75), int((x[0] + x[7]) / 2 + h * 0.75)
            min_digit = 0
            max_right_digit = image.shape[0]
            max_left_digit = image.shape[1]
            x1 = min_digit if x1 < min_digit else x1    # 使用三元运算符改写
            y1 = max_right_digit if y1 > max_right_digit else y1
            x2 = min_digit if x2 < min_digit else x2
            y2 = max_left_digit if y2 > max_left_digit else y2
            digit = image[x1: y1, x2: y2]

            if sum(np.shape(digit)) == 0:
                print(np.shape(digit))
                continue
            hog_trait = st.image2hog(digit)
            if not train_mode:  # 如果开启了训练模式,会读取设定保存的文件目录,然后识别时不经过数字判断
                n_, num = svm_digit_detect(target_num, st.predictShow(svm, hog_trait))
                if not n_:
                    print("wrong digit=", num[0][0])
                    continue
            else:
                st.savetrain(hog_trait, filename=file)
            if armor is not None:
                area.append(armor)
    return area
