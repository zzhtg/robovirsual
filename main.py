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
SerialGive = False
midx = 320
midy = 240
ser = 0
targetnum = 8
recognize_num = 0
key = 0
cv2.
def Stm32(DebugMode = False):
    global key, detect_flag
    while(SerialGive):
        try:
            SerialMsg = ss.Serial_Send(ser, midx, midy)
            if(DebugMode):
                print("You have sent", ord(SerialMsg[0]), SerialMsg[1], SerialMsg[2], SerialMsg[3], ord(SerialMsg[4]), SerialMsg[5], SerialMsg[6], SerialMsg[7], ord(SerialMsg[-1]),"to the serial, x = ", midx, "y = ", midy)
                time.sleep(0.007)
            if(key is ord('q')):
                break 
        except:
            try:
                ser = ss.Serial_init(115200, 1)
                print("Here is no serial device, waiting for a connection...") 
                if(key is ord('q')):
                    break
            except:
                pass

def main(cam, SerialGive = True): 
    """ 
    输入：cam(摄像头选取参数) 功能：主程序 输出：无 
    """ 
    global midx, midy, recognize_num, key, detect_flag 
    svm = cv2.ml.SVM_load('.\\svm_data.dat')
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
        gray, lightGroup = ld.lightDetect(frame, armcolor)
        armorPixel = ad.armorDetect(svm, frame, lightGroup, targetnum)
        naf = pf.putFrameSuccess(frame, runningtime, armorPixel, pf.count)
        if armorPixel:
            for [a, b, x, y] in armorPixel:
                midx = math.ceil((a + x) / 2)
                midy = math.ceil((b + y) / 2)
                pf.showkalman(frame, armorPixel, Matrix, kalman, kp.error, kp.real_error)
        frame = pf.putCrossFocus(frame)
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
    try:
        cam = sys.argv[1]
    except:
        cam = 0
    global t1, t2
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
    else:
        main(cam)
    
    
