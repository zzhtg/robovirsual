# coding=utf-8
import cv2
import numpy as np
import math
import pefermance as pf
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
anglebias = 0
debug_mode = False

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
    //暂时没有采用
    输入：两组 两个(x, y)形式点的信息
    功能：判断两点确定的直线是否平行
    输出：True or False ,斜率差值
    '''
    [lx1, ly1], [lx2, ly2] = p1
    [rx1, ry1], [rx2, ry2] = p2
    lk = (ry1-ly1) / (rx1 - lx1)
    rk = (ry2-ly2) / (rx2 - lx2)
    para = abs((rk - lk) / (1 + lk*rk))
    return para < 0.15, para

def lightDist(ll, lr):
    '''
    输入：左右灯条的像素长度
    功能：计算摄像头到灯条的距离
    输出：比较近的那个灯条的距离
    '''
    llendistance = (547.27 * 5.5) / (1 + ll)
    rlendistance = (547.27 * 5.5) / (1 + lr)
    return min(llendistance, rlendistance)

def orthoPixel(frame, lpixel, rpixel):
    '''
    输入：左右灯条的四个点坐标
    功能：找到左右灯条和中心矢量的对应两个坐标
    输出：中心线、左灯条、右灯条两个坐标
    '''
    lxmid = np.average(lpixel[0:4, 0])
    lymid = np.average(lpixel[0:4, 1])
    rxmid = np.average(rpixel[0:4, 0])
    rymid = np.average(rpixel[0:4, 1])

    ll = squareform(pdist(lpixel, metric='euclidean'))  # 将distA数组变成一个矩阵
    rr = squareform(pdist(rpixel, metric='euclidean'))

    ano_l = np.argsort(ll)   # 对于左灯条的第0个顶点而言，找出能够组成短边的另一个点的序号
    ano_r = np.argsort(rr)  # 对于左灯条的第0个顶点而言，找出能够组成短边的另一个点的序号
    directlxmid, directlymid = (lpixel[0, :] + lpixel[ano_l[0, 1], :]) / 2
    directrxmid, directrymid = (rpixel[0, :] + rpixel[ano_r[0, 1], :]) / 2

    vecmid = [[lxmid, lymid],  [rxmid, rymid]] # 中心连接矢量
    veclightL = [[lxmid, lymid], [directlxmid, directlymid]] # 灯条1的方向矢量
    veclightR = [[rxmid, rymid], [directrxmid, directrymid]] # 灯条2的方向矢量

    if (debug_mode == True):        # debug 划线及输出
        cv2.line(frame, tuple(vecmid[0]), tuple(vecmid[1]), (255, 0, 0), 5)
        cv2.line(frame, tuple(veclightL[0]), tuple(veclightL[1]), (255, 0, 0), 5)
        cv2.line(frame, tuple(veclightR[0]), tuple(veclightR[1]), (255, 0, 0), 5)
    return vecmid, veclightL, veclightR

def orthoAngle(vecmid, veclightL, veclightR):
    '''
    输入：获取左右灯条和中心线的两个坐标
    功能：计算两个灯条的中心连接矢量和两个灯条方向矢量的正交性
    输出：True or False , 左右灯条方向矢量与中心连接矢量的夹角
    '''
    global anglebias
    vecmid = [vecmid[0][i] - vecmid[1][i] for i in range(2)]
    veclightL = [veclightL[0][i] - veclightL[1][i] for i in range(2)]
    veclightR = [veclightR[0][i] - veclightR[1][i] for i in range(2)]

    absC = math.sqrt(vecmid[0]**2 + vecmid[1]**2)
    absL = math.sqrt(veclightL[0]**2 + veclightL[1]**2)
    absR = math.sqrt(veclightR[0]**2 + veclightR[1]**2)

    inl = (vecmid[0] * veclightL[0] + vecmid[1] * veclightL[1]) # 内积
    inr = (vecmid[0] * veclightR[0] + vecmid[1] * veclightR[1])
    inp = (veclightL[0]*veclightR[0] + veclightL[1] * veclightR[1])
    angleL = inl / (absC * absL)  # 左向量与中心向量的夹角
    angleR = inr / (absC * absR)  # 右向量与中心向量的夹角
    angleP = inp / (absL * absR) # 左右向量夹角
    anglebias = (math.atan2(vecmid[0], vecmid[1]) / math.pi * 180.0)
    return_flag = (abs(angleL) < 0.3 and 
            abs(angleR) < 0.3 and
            abs(angleP) > 0.9)
    if return_flag:
        None
    else:
        print(angleL, angleR, angleP)
    if(return_flag and debug_mode):
        print("angleL = ", angleL, "angleR = ", angleR, "midAngle = ", (angleL + angleR) / 2)
    # 范围 60~120度， 两个灯条都满足
    return return_flag, angleL, angleR, angleP

def armorDetect(frame, lightGroup):
    '''
    输入：lightGroup（可能是灯条的矩形最小边界拟合信息）
    功能：一一对比矩形、找到可能的灯条组合作为装甲
    输出：armorArea（可能是装甲的矩形【长宽、左上角坐标,左右灯条长宽平均值】的列表）
    '''
    armorArea = []

    lens = len(lightGroup)
    for left in range(lens):
        for right in range(left + 1, lens):
            if lightGroup[left][0][0] > lightGroup[right][0][0]:
                left, right = right, left
            xL, yL, wL, lL = [j for i in lightGroup[left][0:2] for j in i]
            xR, yR, wR, lR = [j for i in lightGroup[right][0:2] for j in i]
            if wL > lL: 
                wL, lL = lL, wL
            if wR > lR: 
                wR, lR = lR, wR

            l_, lenthDif = lenthDifDet(lL, lR) # 长度差距判断：两灯条的长度差 / 长一点的那个长度 < 36%
            if not l_:
                continue
            w_, widthDif = widthDifDet(wL, wR) # 宽度差距判断：两灯条的宽度差 / 长一点的那个长度 < 68%
            if not w_:
                continue
            a_, armorAspect = armorAspectDet(xL, yL, xR, yR, lL, lR, wL, wR) # 横纵比判断：2.7~4.5
            if not a_:
                continue
            lpixel = cv2.boxPoints(lightGroup[left])
            rpixel = cv2.boxPoints(lightGroup[right])
            #第一个y值最大，第三个y值最小，第二个x值最小，第四个x值最大
            vecmid, veclightL, veclightR = orthoPixel(frame, lpixel, rpixel)
            o_, orthoLValue, orthoRValue, angleP = orthoAngle(vecmid, veclightL, veclightR)# 垂直判断：< 0.9
            if not o_:
                continue

            x = sorted(np.append(lpixel[0:4, 0], rpixel[0:4, 0]))
            y = sorted(np.append(lpixel[0:4, 1], rpixel[0:4, 1]))
            armor = [int(i) for i in [x[0], y[0], x[7], y[7], (wL+wR)/2, (lL+lR)/2]]
            if armor is not None:
                armorArea.append(armor)
    return armorArea