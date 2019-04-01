# coding=utf-8
#/usr/bin/python3
import cv2
import numpy as np
import sys
import armorDetect as ad
import lightDetect as ld
import pefermance as pf
import KalmanPredict as kp 

fps = []        
count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0, 
    'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}

def main(cam):
    """
     输入：cam(摄像头选取参数)
     功能：主程序
     输出：无
     """

    Matrix, kalman = kp.Kalman_init()
    raw_pitch = 0
    raw_yaw = 0
    error = []
    real_error = []

    cap = cv2.VideoCapture(cam)
    #cap.set(3, 1380)
    armcolor = 98  #114: red, 98: blue

    while cap.isOpened():
        t1 = cv2.getTickCount()
        _, frame = cap.read()
        image = frame.copy()
        gray, lightGroup = ld.lightDetect(frame, armcolor)
        armorPixel = ad.armorDetect(frame, lightGroup)

        naf = pf.putFrameSuccess(frame, t1, armorPixel, count)
        pf.showtext(frame, t1, armorPixel, count, fps)
        pf.showkalman(frame, armorPixel, Matrix, kalman, error, real_error)

        if armorPixel:
            for [a, b, x, y, w, h] in armorPixel:
                armor = image[b: y, a: x]
                x1, y1, x2, y2 = int((b+y)/2 - h), int((b+y)/2 + h), int((a+x)/2 - h), int((a+x)/2 + h)
                mindigit = 0
                maxdigit = image.shape[0]
                if x1 < mindigit:
                    x1 = mindigit
                if y1 > maxdigit:
                    y1 = maxdigit
                digit = image[x1: y1, x2: y2]
                digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
                _, digit = cv2.threshold(digit, 10, 255, cv2.THRESH_BINARY)
                cv2.imshow("digit", digit)
        cv2.imshow("frame", frame)



        key = cv2.waitKey(5)
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

    main(cam)
