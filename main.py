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
# import SerialSend as ss    # 串口通信
def main(cam):
    # '''
    #     输入：cam(摄像头选取参数)
    #     功能：主程序
    #     输出：无
    #         '''
    # ser = ss.Serial_init(115200, 1) # get a serial item, first arg is the boudrate, and the second is timeout
    # if(ser == 0):          # if not an existed Serial object 
    #     print("Caution: Serial Not Found!") # print caution
    # else:
    #     global raw_pitch
    #     global raw_yaw
    #     Msg = Process(raw_pitch, raw_yaw)
    #     print("You have sent", ord(Msg[0]), Msg[1:3], ord(Msg[4]), Msg[5:-2], ord(Msg[-1]), 'to the serial\n')
    #     ser.write(Msg.encode('ascii'))
    #     rec = ser.readline()
    #     print(rec)
    #     ser.close()
    
    showimage = True
    showText = True
    onminipc = False
    fps = []        
    count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0, 
            'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}
    armcolor = ord('r')  #114: red, 98: blue
    cap = cv2.VideoCapture(cam)
    cap.set(15, -6)
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
            naf = pf.putMsg(frame, armorPixel, count) #打印信息
            nfps = pf.putFps(frame, t1)
            fps.append(nfps)
            if len(armorPixel) > 0:
                x, y, w, h = armorPixel[0]
                Matrix, error, real_error = kp.Predict(Matrix, kalman, error, real_error, frame, x, y, w, h)
        if showimage:        #如果显示图片
            if len(armorPixel) > 0:
                for x, y, w, h in armorPixel:
                    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                    x, y, w, h = [i+1 for i in [x, y, w, h]]
                    image = frame[y: h, x: w]
                    cv2.imshow("armor", image)
        cv2.imshow("frame", frame)
        if onminipc and naf > 400:       #远程操控妙算按键失效，自动退出
            break

        key = cv2.waitKey(1)

        if key is ord('r') or key is ord('b'):
            armColor = key
        if key is ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    if showText and len(fps):
        print(naf)

if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    try:
        main(cam)
    except:
        main(cam)
