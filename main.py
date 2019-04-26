## /usr/bin/python3
#coding=utf-8
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
import attackjudge as aj
def stm32(debug_mode=False):
    global ser, key, midx, midy, serial_give
    while serial_give:
        try:
            serial_msg = ss.serial_send(ser, midx, midy)
            if debug_mode:
                print("You have sent", ord(serial_msg[0]), serial_msg[1], serial_msg[2], serial_msg[3], ord(serial_msg[4]), serial_msg[5], serial_msg[6], serial_msg[7], ord(serial_msg[-1]),"to the serial, x = ", midx, "y = ", midy)
            time.sleep(0.005)
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
    global midx, midy, frame_x, frame_y
    global key, serial_give, cap, color
    svm = cv2.ml.SVM_load('./svm_data.dat') # 读取svm参数
    kalman = kp.Kalman_Filter() # 卡尔曼滤波器类初始化
    interval = 1.0
    # 配置灯条检测预处理参数
    ld.frame_threshold = [150, 255]             # 二值化阈值
    ld.aspect_threshold = [0.06, 0.5]           # 长宽比阈值
    ld.red_down_threshold = [60, 110, 220]      # 红色阈值下界
    ld.red_up_threshold = [180, 220, 255]       # 红色阈值上界
    ld.blue_down_threshold = [230, 150, 60]     # 蓝色阈值下界
    ld.blue_up_threshold = [255, 250, 150]      # 蓝色阈值上界
    # 配置装甲检测参数
    ad.length_threshold = 1.0                   # 灯条长度比
    ad.width_threshold = 1.0                    # 灯条宽度比
    ad.aspect_threshold = [0.9, 5.0]            # 长宽比
    ad.ortho_threshold = [0.2, 0.2, 0.9]        # 正交率阈值(angle_l,angle_r,angle_p)
    ad.target_num = None                        # 要检测的数字(如果为None表示不加入数字检测条件)
    ad.debug_mode = False						# 正交率划线显示
    ad.error_text = False						# 检测错误输出文本
    # 攻击策略
    while cap.isOpened(): 
        start = time.clock()
        _, frame = cap.read()
        if not EntireWindow: # 全视窗模式 1080p，如果不开的话就是默认截取中间的640x480
            frame = frame[int(frame_y/2)-240: int(frame_y/2)+240, int(frame_x/2)-320: int(frame_x/2)+320]
        gray, group = ld.light_detect(frame, color, preview = True) # preview为显示预处理图像
        # armorgroup = ad.armor_detect(svm, frame, group, train_mode=True, file="F:\\traindata\\"+str(target_num)) # 训练用，需要修改保存训练集目录
        armorgroup = ad.armor_detect(svm, frame, group)
        entire = pf.put_success(frame, interval, armorgroup, pf.count)
        if armorgroup:
            armor = aj.judge(armorgroup, attack = aj.mid, args = None)
            [midx, midy] = armor.mid
            armor.show(frame, kalman, KalmanPreview = True)
            print(armor.digit, armor.dist)
        else:
            midx = 320
            midy = 240

        frame = pf.put_cross_focus(frame, np.shape(frame)[1] / 2, np.shape(frame)[0] / 2) 
        cv2.imshow("frame", frame)

        key = cv2.waitKey(3)
        if key is ord('r') or key is ord('b'):
            color = key
        if key is ord('q'):
            cv2.destroyAllWindows()
            cap.release()   # 摄像头关闭
            break
        interval = time.clock()-start
        # print("timeused = ", 1.0/interval)

if __name__ == "__main__":
    # global variable
    frame_x = 1920  # 全视窗模式分辨率
    frame_y = 1080
    midx = 320      # 非全视窗模式分辨率 
    midy = 240
    color = 98      # 114: red, 98: blue
    ad.debug_mode = True
    serial_give = False
    ser = 0
    key = 0
    # camera load
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_x)
    cap.set(4, frame_y)
    cap.set(15, -8)      # 曝光度最低为-8
    EntireWindow = False # 全视窗模式
    # if or if not serial 
    if serial_give:
        ser = ss.serial_init(115200, 1)  # get a serial item, first arg is the boudrate, and the second is timeout
        if ser == 0:  # if not an existed Serial object
            print("Caution: Serial Not Found!")  # print caution
        thread_main = threading.Thread(target=main)
        thread_serial = threading.Thread(target=stm32, args=(serial_give,))
        thread_main.start()
        thread_serial.start()
        thread_main.join()
        thread_serial.join()
        print("Threads has been stopped")
    else: # None Serial Debug Connection
        main()
