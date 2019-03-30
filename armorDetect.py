# coding=utf-8
import cv2
import numpy as np
import math
import pefermance as pf

def lenthDifDet(lLeft, lRight):
    '''
    输入：lLeft（左灯条长度）、lRight（右灯条长度）
    功能：检测当前灯条组合是否符合长度差距条件
    输出：True 或者 False 以及 长度宽度比 '''
    lenthDif = abs(lLeft-lRight) / max(lLeft, lRight)
    return  lenthDif <= 0.36, lenthDif

def widthDifDet(wLeft, wRight):
    '''
    输入：wLeft（左灯条宽度）wRight（右灯条宽度）
    功能：检测当前灯条组合是否符合宽度差距条件
    输出：True 或者 False 以及 宽度差距比
    '''
    wLeft, wRight = [i+1 for i in [wLeft, wRight]]
    widthDif = abs(wLeft-wRight) / max(wLeft, wRight)
    return widthDif <= 0.68, widthDif

def armorAspectDet(xL, yL, xR, yR, lL, lR, wL, wR):
    '''
    输入：最小矩形拟合信息（x、y、w、h）L（左灯条）（x、y、w、h、）R（右灯条）
    功能：检测当前灯条组合是否符合横纵比条件
    输出：True 或者 False 以及 装甲横纵比
    '''
    armorAspect = math.sqrt((yR-yL)**2 + (xR-xL)**2) / max(lL, lR, wL, wR)
    return (4.5 >= armorAspect and armorAspect >= 1.7), armorAspect

def paralle(p1, p2):
    '''
    输入：两组 两个(x, y)形式点的信息
    功能：判断两点确定的直线是否平行
    输出：True or False ,斜率差值
    '''
    [lx1, ly1], [lx2, ly2] = p1
    [rx1, ry1], [rx2, ry2] = p2
    lk = (ry1-ly1) / (rx1 - lx1)
    rk = (ry2-ly2) / (rx2 - lx2)
    paralle = abs((rk - lk) / (1 + lk*rk))
    return paralle < 0.15, paralle

def lightDist(ll, lr):
    '''
    输入：左右灯条的像素长度
    功能：计算摄像头到灯条的距离
    输出：比较近的那个灯条的距离
    '''
    llendistance = (547.27 * 5.5) / (1 + ll)
    rlendistance = (547.27 * 5.5) / (1 + lr)
    return min(llendistance, rlendistance)

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

            xL, yL, wL, lL = [j for i in lightGroup[left][0:2] for j in i]
            xR, yR, wR, lR = [j for i in lightGroup[right][0:2] for j in i]
            #pf.putInfo(frame, int(xL), int(yL), lightDist(lL, lR), "lightDist")
            if wL > lL: 
                wL, lL = lL, wL
            if wR > lR: 
                wR, lR = lR, wR

            l_, lenthDif = lenthDifDet(lL, lR)
            if not l_:
                print("lenthDif", lenthDif)
                continue
            w_, widthDif = widthDifDet(wL, wR)
            if not w_:
                print("widthDif", widthDif)
                continue
            a_, armorAspect = armorAspectDet(xL, yL, xR, yR, lL, lR, wL, wR)
            if not a_:
                print("armorAspect", armorAspect)
                continue
            lpixel = cv2.boxPoints(lightGroup[left])
            rpixel = cv2.boxPoints(lightGroup[right])
            #第一个y值最大，第三个y值最小，第二个x值最小，第四个x值最大
            p_, paraValue = paralle([lpixel[0], lpixel[2]], 
                                    [rpixel[0], rpixel[2]])
            if not p_:
                print("paraValue", paraValue)
                continue

            x = sorted(np.append(lpixel[0:4, 0], rpixel[0:4, 0]))
            y = sorted(np.append(lpixel[0:4, 1], rpixel[0:4, 1]))
            armor = [int(i) for i in [x[0], y[0], x[7], y[7]]]
            if armor is not None:
                armorArea.append(armor)
    return armorArea

