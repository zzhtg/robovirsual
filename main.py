#/usr/bin/python3
import cv2
import numpy as np
import sys
from armorDetect import *
from lightDetect import *
from pefermance import *

def main(cam):
    '''
    输入：cam(摄像头选取参数)
    功能：主程序
    输出：无
    '''
    fps = []
    count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0, 
            'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}

    armcolor = ord('r')  #114: red, 98: blue
    cv2.namedWindow("main")
    cap = cv2.VideoCapture(cam)
    cap.set(15, -6)
    #cap.set(3, 1380)

    while cap.isOpened():
        t1 = cv2.getTickCount()
        _, frame = cap.read()
        lightGroup = lightDetect(frame, armcolor)
        armorPixel = armorDetect(frame, lightGroup)

        naf = putMsg(frame, armorPixel, count) #打印信息
        fps.append(putFps(frame, t1))

        key = cv2.waitKey(10)
        if key is ord('r') or key is ord('b'):
            armColor = key
        if key is ord('q'):
            break
#        if naf > 400:       #远程操控妙算按键失效，三百帧后自动退出
#            break           #在电脑上使用的时候可以注释掉这两行代码

    if len(fps):    #打印信息
        print(naf)
        FpsTimeHist(fps)

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    main(cam)
