# /usr/bin/python3
# coding=utf-8
# Official Package
import sys
import os
import math
import time
import threading
import cv2
import numpy as np

# Private Function Package
import armorDetect as ad
import lightDetect as ld
import pefermance as pf
import Kalmanpredict as kp
import SerialSend as ss
import SvmTrain as st


ad.debug_mode = False
serial_give = False
midx = 320
midy = 240
ser = 0
target_num = 8
recognize_num = 0
key = 0


def stm32(debug_mode=False):
    global key, detect_flag
    while serial_give:
        try:
            serial_msg = ss.Serial_Send(ser, midx, midy)
            if debug_mode:
                print("You have sent", ord(serial_msg[0]), serial_msg[1], serial_msg[2], serial_msg[3], ord(serial_msg[4]), serial_msg[5], serial_msg[6], serial_msg[7], ord(serial_msg[-1]),"to the serial, x = ", midx, "y = ", midy)
                time.sleep(0.007)
            if key is ord('q'):
                break 
        except:
            try:
                ser = ss.serial_init(115200, 1)
                print("Here is no serial device, waiting for a connection...") 
                if key is ord('q') :
                    break
            except:
                pass


def main(cam, serial_give = True): 
    """ 
    输入：cam(摄像头选取参数) 功能：主程序 输出：无 
    """ 
    global midx, midy, recognize_num, key, detect_flag 
    svm = cv2.ml.SVM_load('.\\svm_data.dat')
    matrix, kalman = kp.kalman_init()
    cap = cv2.VideoCapture(cam) 
    # cap.set(3, 1380)     
    color = 98  # 114: red, 98: blue 
    interval = 1.0 
    while cap.isOpened(): 
        start = time.clock()
        midx = 320
        midy = 240
        _, frame = cap.read()
        gray, group = ld.lightDetect(frame, color)
        coordinate = ad.armorDetect(svm, frame, group, target_num)
        entire = pf.put_success(frame, interval, coordinate, pf.count)
        if coordinate:
            for [a, b, x, y] in coordinate:
                midx = math.ceil((a + x) / 2)
                midy = math.ceil((b + y) / 2)
                pf.show_kalman(frame, coordinate, matrix, kalman, kp.error, kp.real_error)
        frame = pf.put_cross_focus(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(10)

        if key is ord('r') or key is ord('b'):
            color = key
        if key is ord('q'):
            cv2.destroyAllWindows()
            cap.release()  # 摄像头关闭
            break
        end = time.clock()
        interval = end-start


if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0
    global t1, t2
    if serial_give:
        ser = ss.serial_init(115200, 1)  # get a serial item, first arg is the boudrate, and the second is timeout
        if ser == 0:  # if not an existed Serial object
            print("Caution: Serial Not Found!")  # print caution
        t1 = threading.Thread(target=main, args=(cam, serial_give,))
        t2 = threading.Thread(target=stm32, args=(serial_give,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        print("Threads has been stopped")
    else:
        main(cam)
