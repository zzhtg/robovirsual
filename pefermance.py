# coding=utf-8
import cv2
import numpy as np

def putFps(frame, e1):
    '''
    输入：frame(当前帧）、t1（时间起点）
    功能：添加FPS（Frames Per Second）信息并显示当前帧
    输出：无
    '''
    fps = 1.0 / ((cv2.getTickCount() - e1) / cv2.getTickFrequency())
    fpsstr = "FPS:{0:0.1f}".format(fps)
    cv2.putText(frame, fpsstr, (50, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    cv2.imshow("main", frame)
    return fps

def putMsg(frame, armor, count):
    '''
    输入：frame(当前帧)、 armor(装甲列表)、count(计数成员字典)
    功能：当前帧添加实时成功率与全过程成功率、画出装甲图像
    输出：无
    '''
    count['alFrame'] += 1
    count['perFrame'] += 1
    count['alSucRatio'] = count['alSuc'] / count['alFrame']
    if len(armor) > 0:
        for x, y, w, h in armor:
            cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
            x, y, w, h = [i+1 for i in [x, y, w, h]]
            image = frame[y: h, x: w]
            cv2.imshow("armor", image)
            #print(x,y,w,h)
        count['alSuc'] += 1
        count['perSuc'] += 1
    if count['alFrame'] % count['period'] is 0:
        count['perSucRatio']  = count['perSuc'] / count['perFrame']
        count['perFrame'] = count['perSuc'] = 0
    font = cv2.FONT_ITALIC
    massege = "intime:{0:0.2f}  alltime:{1:0.2f}".format(count['perSucRatio'], count['alSucRatio'])
    cv2.putText(frame, massege, (250, 50), font, 0.8, (255, 255, 255), 2)
    return count['alFrame']

def FpsTimeHist(fps):
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

def putInfo(frame, x, y, *arg, font = cv2.FONT_ITALIC, k = 15):
    for i, info in enumerate(arg):
        cv2.putText(frame, "{0}".format(info), (x, y - k*(i+1)), font, 0.4, (0, 255, 0), 1)
