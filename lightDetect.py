# coding=utf-8
import cv2
import numpy as np


def frameReady(image):
    '''
    输入：image(当前帧图像)
    功能：灰度、二值化、膨胀再腐蚀等处理
    输出：image（处理后图像）
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(cv2.dilate(image, kernel, iterations = 1), kernel, iterations = 1)
    return image

def lightAspectDet(lightRectangle):
    '''
    输入：rectangle(灯条最小拟合矩形数据)
    功能：检测灯条边界矩形的横纵比是否符合条件
    输出：True 或者 False、aspect（灯条横纵比）
    '''
    w, h = lightRectangle[1]
    w += 1
    h += 1
    if w > h:
        w, h = h, w
    aspect = w/h
    return aspect < 0.3 and aspect > 0.06, aspect

def aimColormean(lightArea, mask, armcolor):
    '''
    输入：lightArea(灯条最小拟合矩形图像)、mask(灯条发光区域)、armcolor(装甲颜色)
    功能：检测灯条颜色是否符合条件（BGR颜色空间当前红色或蓝色分量是否够大）
    输出：meanVal(True 或者 False)、mean_val(颜色空间平均值)
    '''
    mean_val = cv2.mean(lightArea, mask)
    limit = 200
    #print("mean_val: ", mean_val)
    if armcolor is 114:
        meanVal = mean_val[2] > limit and mean_val[2] > mean_val[0]
    if armcolor is 98:
        meanVal = mean_val[0] > limit and mean_val[0] > mean_val[2]
    return meanVal, mean_val

def lightDetect(image, armcolor):
    '''
    输入：image(当前帧图像)、armcolor(装甲颜色)
    功能：找到符合颜色、横纵比条件的类似灯条的矩形
    输出：符合条件的矩形（最小边界拟合信息）的列表，不一定就是灯条
    '''
    lightGroup = []
    readyDst = frameReady(image)
    font = cv2.FONT_ITALIC
#opencv 4.0.0 finContours返回轮廓和层级
#opencv 3 finContours 返回图像、轮廓和层级
#修改1 为 0即可
    for contour in cv2.findContours(readyDst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]:
        lightRectangle = cv2.minAreaRect(contour)
        [x, y], [w, h] = lightRectangle[0], lightRectangle[1]
        l_, aspect = lightAspectDet(lightRectangle)
        if not l_:
            #aspectMassege = "aspect{0:.1f}-{1:.1f}-{2:.1f}".format(aspect, w, h)
            #cv2.putText(image, aspectMassege, (int(x), int(y)), font, 0.4, (0, 255, 0), 1)
            continue

        x, y, w, h = cv2.boundingRect(contour)
        lightArea = image[y: y+h, x: x+w]
        mask = readyDst[y: y+h, x: x+w]
        lightArea = cv2.bitwise_and(lightArea, lightArea, mask = mask)
        c_, mean_val = aimColormean(lightArea, mask, armcolor)
        if not c_:
            #massege = "mean_val{0:.0f},{1:.0f},{2:.0f}".format(mean_val[0], mean_val[1], mean_val[2])
            #cv2.putText(image, massege, (x, y-15), font, 0.4, (0, 255, 0), 1)
            continue
        #cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        lightGroup.append(lightRectangle)
    return image, lightGroup

