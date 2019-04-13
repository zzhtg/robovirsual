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
import KalmanPredict as kp
import SerialSend as ss
import SvmTrain as st

def stm32(debug_mode=False):
    global key, midx, midy, serial_give
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


def main():
    """ 
    输入：cam(摄像头选取参数) 功能：主程序 输出：无 
    """ 
    global midx, midy, recognize_num, frame_x, frame_y
    global serial_give, cap, color
    svm = cv2.ml.SVM_load('./svm_data.dat')
    matrix, kalman = kp.kalman_init()
    interval = 1.0

    while cap.isOpened(): 
        start = time.clock()
        midx = frame_x / 2 # 默认发中心坐标的值，如果没有识别到
        midy = frame_y / 2
        _, frame = cap.read()
        frame = frame[midx-320: midx+320, midy-240: midy+320]
        gray, group = ld.light_detect(frame, color)
        coordinate = ad.armor_detect(svm, frame, group, target_num)
        entire = pf.put_success(frame, interval, coordinate, pf.count)
        if coordinate:
            for [a, b, x, y] in coordinate:
                midx = math.ceil((a + x) / 2)
                midy = math.ceil((b + y) / 2)
                pf.show_kalman(frame, coordinate, matrix, kalman, kp.error, kp.real_error)
        frame = pf.put_cross_focus(frame, frame_x / 2, frame_y / 2)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(3)

        if key is ord('r') or key is ord('b'):
            color = key
        if key is ord('q'):
            cv2.destroyAllWindows()
            cap.release()   # 摄像头关闭
            break
        interval = time.clock()-start
        print("timeused = ", 1.0/interval)


if __name__ == "__main__":
    # global variable
    frame_x = 1920
    frame_y = 1080
    color = 98  # 114: red, 98: blue
    ad.debug_mode = False
    serial_give = False
    ser = 0
    target_num = 3
    recognize_num = 0
    key = 0
    # camera load
    cap = cv2.VideoCapture(1)
    cap.set(3, frame_x)
    cap.set(4, frame_y)
    # cap.set(15, -7)
    # if or if not serial 
    if serial_give:
        ser = ss.serial_init(115200, 1)  # get a serial item, first arg is the boudrate, and the second is timeout
        if ser == 0:  # if not an existed Serial object
            print("Caution: Serial Not Found!")  # print caution
        thread_main = threading.Thread(target=main, args=(cam, serial_give,))
        thread_serial = threading.Thread(target=stm32, args=(serial_give,))
        thread_main.start()
        thread_serial.start()
        thread_main.join()
        thread_serial.join()
        print("Threads has been stopped")
    else: # None Serial Debug Connection
        main()
