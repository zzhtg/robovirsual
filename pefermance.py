# coding=utf-8
import cv2
import numpy as np
import KalmanPredict as kp
count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0,
    'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}
fps = []
def putFps(frame, e1):
    '''
    输入：frame(当前帧）、t1（时间起点）
    功能：添加FPS（Frames Per Second）信息并显示当前帧
    输出：无
    '''
    return 1.0 / ((cv2.getTickCount() - e1) / cv2.getTickFrequency())

def putFrameSuccess(frame, e1, armor, count):
    '''
    输入：frame(当前帧)、 armor(装甲列表)、count(计数成员字典)
    功能：当前帧添加实时成功率与全过程成功率、画出装甲图像
    输出：无
    '''
    count['alFrame'] += 1
    count['perFrame'] += 1
    count['alSucRatio'] = count['alSuc'] / count['alFrame']
    if armor:
        count['alSuc'] += 1
        count['perSuc'] += 1
    if count['alFrame'] % count['period'] is 0:
        count['perSucRatio']  = count['perSuc'] / count['perFrame']
        count['perFrame'] = count['perSuc'] = 0
    font = cv2.FONT_ITALIC
    fps = putFps(frame, e1)
    massege = "fps:{0:0.2f} intime:{1:0.2f}  alltime:{2:0.2f}".format(fps, count['perSucRatio'], count['alSucRatio'])
    cv2.putText(frame, massege, (50, 50), font, 0.8, (255, 255, 255), 2)
    return count['alFrame']

def fpsCount(fps):
    """
    输入：fps(包含fps信息的列表)
    功能：画出每帧对应的fps，显示执行过程当中最大、最小和平均帧率
    输出：无
    """
    fps.sort()
    fps[0:] = fps[1:]
    sumFps = 0
    for num in fps:
        sumFps += num
    print("average fps = {0:0.1f}".format(sumFps / len(fps)))
    print("max fps = {0:0.1f}".format(fps[len(fps) - 1]))
    print("min fps = {0:0.1f}".format(fps[0]))


def showtext(frame, t1, armorPixel, count, fps):
    nfps = putFps(frame, t1)
    fps.append(nfps)

def showkalman(frame, armorPixel, Matrix, kalman, error, real_error):
    if armorPixel:
        x, y, w, h = armorPixel[0][0:4]
        Matrix, error, real_error = kp.Predict(Matrix, kalman, error, real_error, frame, x, y, w, h)

