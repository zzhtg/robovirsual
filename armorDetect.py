import cv2
import numpy as np
import math


def paralle(left, right):
    [lx1, ly1], [lx2, ly2] = left[0], left[2]
    [rx1, ry1], [rx2, ry2] = right[0], right[2]
    lk = (ry1-ly1) / (rx1 - lx1)
    rk = (ry2-ly2) / (rx2 - lx2)
    paralle = abs((rk - lk) / (1 + lk*rk))
    return paralle < 0.05, paralle


def armorPixel(leftLight, rightLight):
    '''
    输入：最小边界拟合矩形leftLight（左灯条）、rightLight（右灯条）
    功能：检查平行、获取装甲的左上角坐标以及右下角坐标
    输出：装甲左上角、右下角坐标（方便取出装甲图像）
    '''
    lpixel = cv2.boxPoints(leftLight[0:3])
    rpixel = cv2.boxPoints(rightLight[0:3])
    x = sorted(np.append(lpixel[0:4, 0], rpixel[0:4, 0]))
    y = sorted(np.append(lpixel[0:4, 1], rpixel[0:4, 1]))
    if paralle(lpixel, rpixel):
        return [int(i) for i in [x[0], y[0], x[7], y[7]]]

def lenthDifDet(lLeft, lRight):
    '''
    输入：lLeft（左灯条长度）、lRight（右灯条长度）
    功能：检测当前灯条组合是否符合长度差距条件
    输出：True 或者 False 以及 长度宽度比 '''
    lenthDif = abs(lLeft-lRight) / max(lLeft, lRight)
    return  lenthDif <= 0.23, lenthDif

def widthDifDet(wLeft, wRight):
    '''
    输入：wLeft（左灯条宽度）wRight（右灯条宽度）
    功能：检测当前灯条组合是否符合宽度差距条件
    输出：True 或者 False 以及 宽度差距比
    '''
    widthDif = abs(wLeft-wRight) / max(wLeft, wRight)
    return widthDif <= 0.68, widthDif

def armorAspectDet(xLeft, yLeft, xRight, yRight, lLeft, lRight, wLeft, wRight):
    '''
    输入：最小矩形拟合信息（x、y、w、h）Left（左灯条）（x、y、w、h、）Right（右灯条）
    功能：检测当前灯条组合是否符合横纵比条件
    输出：True 或者 False 以及 装甲横纵比
    '''
    armorAspect = math.sqrt((yRight-yLeft)**2 + (xRight-xLeft)**2) / max(lLeft, lRight, wLeft, wRight)
    return ((7.5 >= armorAspect and armorAspect >= 6) or 
            (3.5 >= armorAspect and armorAspect >= 1.9)), armorAspect

def isArmor(frame, leftLight, rightLight):
    '''
    输入：矩形最小拟合信息——leftLight（左灯条）、rightLight（右灯条）
    功能：检测当前的灯条组合是否满足（组合后横纵比、灯条之间高度差、灯条之间的宽度差距）条件
    输出：True 或者 False
    '''
    k = 15
    [xLeft, yLeft], [wLeft, lLeft] = leftLight[0:2]
    [xRight, yRight], [wRight, lRight] = rightLight[0:2]
    if wLeft > lLeft: 
        wLeft, lLeft = lLeft, wLeft
    if wRight > lRight: 
        wRight, lRight = lRight, wRight

    l_, lenthDif = lenthDifDet(lLeft, lRight)
    w_, widthDif = widthDifDet(wLeft, wRight)
    a_, armorAspect = armorAspectDet(xLeft, yLeft, xRight, yRight, lLeft, lRight, wLeft, wRight)

    cv2.putText(frame, "{0:.1f} {1:.1f}".format(lLeft, wLeft), (
        int(xLeft), int(yLeft) - k*4), cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, "{0:.1f} {1:.1f}".format(lRight, wRight), (
        int(xRight), int(yRight) - k*3), cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)
    if not l_:
        cv2.putText(frame, "lenthDif:{0:.2f}".format(lenthDif), (
            int(xLeft), int(yLeft) + k*2), cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)
    if not w_:
        cv2.putText(frame, "widthDif{0:.2f}".format(widthDif), (
            int(xLeft), int(yLeft) + k*3), cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)
    if not a_:
        cv2.putText(frame, "armorAspect{0:.2f}".format(armorAspect), (
            int(xLeft), int(yLeft) + k*4), cv2.FONT_ITALIC, 0.4, (0, 255, 0), 1)
    return w_ and l_ and a_

def armorDetect(frame, lightGroup):
    '''
    输入：lightGroup（可能是灯条的矩形最小边界拟合信息）
    功能：一一对比矩形、找到可能的灯条组合作为装甲
    输出：armorArea（可能是装甲的矩形【长宽、左上角坐标】的列表）
    '''
    armorArea = []

    lens = len(lightGroup)
    for left in range(lens):
        for right in range(left + 1, lens):
            if lightGroup[left][0][0] > lightGroup[right][0][0]:
                left, right = right, left
            if not isArmor(frame, lightGroup[left], lightGroup[right]):
                continue
            armor = armorPixel(lightGroup[left], lightGroup[right])
            if armor is not None:
                armorArea.append(armor)
    return armorArea

