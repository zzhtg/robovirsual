#/usr/bin/python3
# coding=utf-8
# Official Package
import cv2
import numpy as np
import sys
import os
import math
import time
import threading
# Private Funtion Package
import armorDetect as ad
import lightDetect as ld
import pefermance as pf
import KalmanPredict as kp
import SerialSend as ss
import SvmTrain as st
ad.debugmode = False
SerialGive = True
midx = 320
midy = 240
ser = 0
targetnum = 6
recognize_num = 0
key = 0
def Stm32(DebugMode = True):
    global key
    while(SerialGive):
        SerialMsg = ss.Serial_Send(ser, midx, midy)
        if(DebugMode):
            print("You have sent %d %s %d %s %d to the serial, x = %d, y = %d\n"%
                (ord(SerialMsg[0]), SerialMsg[1:3], ord(SerialMsg[4]),
                SerialMsg[5:-2], ord(SerialMsg[-1]), midx, midy))
        if(key is ord('q')):
            break    
        
def main(cam, SerialGive = True):
    """
     输入：cam(摄像头选取参数)
     功能：主程序
     输出：无
     """
    global midx, midy, recognize_num, key
    svm = cv2.ml.SVM_load('/home/ubuntu/robovirsual/svm_data.dat')
    Matrix, kalman = kp.Kalman_init()
    cap = cv2.VideoCapture(cam)
    # cap.set(3, 1380)
    armcolor = 98  # 114: red, 98: blue
    runningtime = 1.0
    while cap.isOpened():
        start = time.clock()
        midx = 320
        midy = 240
        _, frame = cap.read()
        image = frame.copy()
        gray, lightGroup = ld.lightDetect(frame, armcolor)
        armorPixel = ad.armorDetect(frame, lightGroup)
        naf = pf.putFrameSuccess(frame, runningtime, armorPixel, pf.count)
        if(targetnum == recognize_num):
            pf.showkalman(frame, armorPixel, Matrix, kalman, kp.error, kp.real_error)
            print(recognize_num)
        if armorPixel:
            for [a, b, x, y, w, h] in armorPixel:
                midx = math.ceil((a + x) / 2)
                midy = math.ceil((b + y) / 2)
                armor = image[b: y, a: x]
                x1, y1, x2, y2 = int((b+y)/2 - h), int((b+y)/2 + h), int((a+x)/2 - h * 0.75), int((a+x)/2 + h * 0.75)
                mindigit = 0
                maxdigit = image.shape[0]
                if x1 < mindigit:
                    x1 = mindigit
                if y1 > maxdigit:
                    y1 = maxdigit
                digit = image[x1: y1, x2: y2]
                hogtrait = st.image2hog(digit)
                # st.savetrain(hogtrait, filename = "F:\\traindata")
                recognize_num = st.PredictShow(svm, hogtrait)[0][0]

        cv2.imshow("frame", frame)
        key = cv2.waitKey(10)
        if key is ord('r') or key is ord('b'):
            armcolor = key
        if(key is ord('q')):
            cv2.destroyAllWindows()
            cap.release()  # 摄像头关闭
            break
        end = time.clock()
        runningtime = end-start

if __name__ == "__main__":
    global t1, t2
    try:
        cam = sys.argv[1]
    except:
        cam = 0
    if (SerialGive):
        ser = ss.Serial_init(115200, 1)  # get a serial item, first arg is the boudrate, and the second is timeout
        if (ser == 0):  # if not an existed Serial object
            print("Caution: Serial Not Found!")  # print caution
    t1 = threading.Thread(target=main, args=(cam, SerialGive,))
    t2 = threading.Thread(target=Stm32, args=(SerialGive,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("Threads has been stopped")
    #main(cam)
    
    
