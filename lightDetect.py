# coding=utf-8
import cv2
import numpy as np


def frame_ready(image):
    """
    输入：image(当前帧图像)
    功能：灰度、二值化、膨胀再腐蚀等处理
    输出：image（处理后图像）
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(cv2.dilate(image, kernel, iterations=1), kernel, iterations=1)
    return image


def light_aspect_det(rect):
    """
    输入：rectangle(灯条最小拟合矩形数据)
    功能：检测灯条边界矩形的横纵比是否符合条件
    输出：True 或者 False、aspect（灯条横纵比）
    """
    w, h = rect[1]
    w += 1
    h += 1
    if w > h:
        w, h = h, w
    aspect = w/h
    return 0.06 < aspect < 0.5, aspect


def aim_color_mean(area, mask, color):
    """
    输入：area(灯条最小拟合矩形图像)、mask(灯条发光区域)、color(装甲颜色)
    功能：检测灯条颜色是否符合条件（BGR颜色空间当前红色或蓝色分量是否够大）
    输出：_c(True 或者 False)、value(颜色空间平均值)
    """
    value = cv2.mean(area, mask)
    _c = False
    if color is 114:
        _c = (60 <= value[0] <= 160, 150 <= value[1] <= 220, 240 <= value[2] <= 255)
    if color is 98:
        _c = (240 <= value[0] <= 255, 150 <= value[1] <= 250, 60 <= value[2] <= 240)
    return _c, value


def light_detect(image, color):
    """
    输入：image(当前帧图像)、color(装甲颜色)
    功能：找到符合颜色、横纵比条件的类似灯条的矩形
    输出：符合条件的矩形（最小边界拟合信息）的列表，不一定就是灯条
    """
    group = []
    pretreatment = frame_ready(image)
    font = cv2.FONT_ITALIC
    # cv2 4.0.0 finContours返回轮廓和层级
    # cv2 3 finContours 返回图像、轮廓和层级
    # 修改1 为 0即可
    for contour in cv2.findContours(pretreatment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        rect = cv2.minAreaRect(contour)
        [x, y], [w, h] = rect[0], rect[1]
        l_, aspect = light_aspect_det(rect)
        if not l_:
            msg = "aspect{0:.1f}-{1:.1f}-{2:.1f}".format(aspect, w, h)
            # cv2.putText(image, msg, (int(x), int(y)), font, 0.4, (0, 255, 0), 1)
            continue

        x, y, w, h = cv2.boundingRect(contour)
        area = image[y: y+h, x: x+w]
        mask = pretreatment[y: y+h, x: x+w]
        area = cv2.bitwise_and(area, area, mask=mask)
        c_, value = aim_color_mean(area, mask, color)
        if False in c_:
            msg = "value{0:.0f},{1:.0f},{2:.0f}".format(value[0], value[1], value[2])
            # cv2.putText(image, msg, (x, y-15), font, 0.4, (0, 255, 0), 1)
            continue
        group.append(rect)
    return pretreatment, group
