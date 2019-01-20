#/usr/bin/python3
import cv2
import numpy as np
import sys
from armorDetect import *
from lightDetect import *
from measurement import *


def main(cam):
    '''
    输入：cam(摄像头选取参数)
    功能：主程序
    输出：无
    '''
    cap = cv2.VideoCapture(cam)
    cap.set(15, -4)
    #cap.set(3, 1380)
    armcolor = ord('r')  #114: red, 98: blue
    cv2.namedWindow("main")
    count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0, 
            'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}

    while cap.isOpened():
        t1 = cv2.getTickCount()
        _, frame = cap.read()
        lightGroup = lightDetect(frame, armcolor)
        armorPixel = armorDetect(frame, lightGroup)

        naf = putMsg(frame, armorPixel, count)
        measurement(frame, t1)
        key = cv2.waitKey(7)
        if key is ord('r') or key is ord('b'):
            armColor = key
        if key is ord('q'):
            break
#        if naf > 300:       #远程操控妙算按键失效，三百帧后自动退出
#            break           #在电脑上使用的时候可以注释掉这两行代码

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    main(cam)
