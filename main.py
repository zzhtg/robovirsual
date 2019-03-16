# coding=utf-8
#/usr/bin/python3
import cv2
import numpy as np
import sys
import armorDetect as ad
import lightDetect as ld
import pefermance as pf
import KalmanPredict as kp
# import threading           # 多线程
import SerialSend as ss    # 串口通信

showimage = True
showText = True
onminipc = False
fps = []        
count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0, 
    'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}

def main(cam):
    # '''
    #     输入：cam(摄像头选取参数)
    #     功能：主程序
    #     输出：无
    #         '''
    raw_pitch = 0
    raw_yaw = 0
    ser = ss.Serial_init(115200, 1) # get a serial item, first arg is the boudrate, and the second is timeout
    if(ser == 0):          # if not an existed Serial object 
        print("Caution: Serial Not Found!") # print caution
    armcolor = ord('r')  #114: red, 98: blue
    cap = cv2.VideoCapture(cam)
    #cap.set(3, 1380)
    Matrix, kalman = kp.Kalman_init()
    error = []
    real_error = []
    while cap.isOpened():
        t1 = cv2.getTickCount()
        _, frame = cap.read()
        gray, lightGroup = ld.lightDetect(frame, armcolor)
        armorPixel = ad.armorDetect(frame, lightGroup)

        if showText:     #如果显示文本
            naf = pf.putMsg(frame, t1, armorPixel, count) #打印信息
            nfps = pf.putFps(frame, t1)
            fps.append(nfps)
            if len(armorPixel) > 0:
                x, y, w, h = armorPixel[0]
                Matrix, error, real_error = kp.Predict(Matrix, kalman, error, real_error, frame, x, y, w, h)
                print("x = %d, y = %d"%(x, y))
                Text_Send, Text_Read = ss.Serial_Send(ser, x, y)
                print("Send:%s\n%s"%(Text_Send, Text_Read))
        if showimage:        #如果显示图片
            if len(armorPixel) > 0:
                for x, y, w, h in armorPixel:
                    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                    x, y, w, h = [i+1 for i in [x, y, w, h]]
                    image = frame[y: h, x: w]
                    cv2.imshow("armor", image)
            cv2.imshow("frame", frame)
        if onminipc and naf > 200:       #远程操控妙算按键失效，自动退出
            break

        key = cv2.waitKey(5)
        if key is ord('r') or key is ord('b'):
            armColor = key
        if key is ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release() # 摄像头关闭
    ser.close()   # 串口关闭
    if showText and len(fps):
        print(naf)

if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    main(cam)
