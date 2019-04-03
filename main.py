# coding=utf-8
#/usr/bin/python3
# Official Package
import cv2
import numpy as np
import sys
import os
import math
import time
# Private Funtion Package
import armorDetect as ad
import lightDetect as ld
import pefermance as pf
import KalmanPredict as kp
import SerialSend as ss
import SvmTrain as svm
ad.debugmode = False
def main(cam, SerialGive = True):
    """
     输入：cam(摄像头选取参数)
     功能：主程序
     输出：无
     """
    if(SerialGive):
        ser = ss.Serial_init(115200, 1)  # get a serial item, first arg is the boudrate, and the second is timeout
        if (ser == 0):  # if not an existed Serial object
            print("Caution: Serial Not Found!")  # print caution
    Matrix, kalman = kp.Kalman_init()
    cap = cv2.VideoCapture(cam)
    #cap.set(3, 1380)
    armcolor = 98  #114: red, 98: blue

    while cap.isOpened():
        t1 = cv2.getTickCount()
        _, frame = cap.read()
        image = frame.copy()
        gray, lightGroup = ld.lightDetect(frame, armcolor)
        armorPixel = ad.armorDetect(frame, lightGroup)

        naf = pf.putFrameSuccess(frame, t1, armorPixel, pf.count)
        pf.showtext(frame, t1, armorPixel, pf.count, pf.fps)
        pf.showkalman(frame, armorPixel, Matrix, kalman, kp.error, kp.real_error)

        if armorPixel:
            for [a, b, x, y, w, h] in armorPixel:
                midx = math.ceil((a + x) / 2)
                midy = math.ceil((b + y) / 2)
                # print("x = %d, y = %d" % (midx, midy))
                if(SerialGive):
                    Text_Send, Text_Read = ss.Serial_Send(ser, midx, midy)
                    print("Send:%s\n%s" % (Text_Send, Text_Read))
                armor = image[b: y, a: x]
                x1, y1, x2, y2 = int((b+y)/2 - h), int((b+y)/2 + h), int((a+x)/2 - h * 0.75), int((a+x)/2 + h * 0.75)
                mindigit = 0
                maxdigit = image.shape[0]
                if x1 < mindigit:
                    x1 = mindigit
                if y1 > maxdigit:
                    y1 = maxdigit
                digit = image[x1: y1, x2: y2]
                # svm.savetrain(digit, filename = "F:\\traindata")

        cv2.imshow("frame", frame)
        key = cv2.waitKey(10)
        if key is ord('r') or key is ord('b'):
            armcolor = key
        if key is ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release() # 摄像头关闭
    if pf.showtext and len(fps):
        print(naf)

if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0
    main(cam, SerialGive = False)
