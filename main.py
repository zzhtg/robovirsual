# coding=utf-8
#/usr/bin/python3
import cv2
import numpy as np
import sys
import armorDetect as ad
import lightDetect as ld
import pefermance as pf

def main(cam):
    '''
    输入：cam(摄像头选取参数)
    功能：主程序
    输出：无
    '''
    fps = []
    showimage = True
    showText = True
    count = {'perSucRatio':0, 'alSucRatio':0, 'alFrame':0, 
            'alSuc':0, 'perFrame':0, 'perSuc':0, 'period':30}

    armcolor = ord('r')  #114: red, 98: blue
    cap = cv2.VideoCapture(cam)
    cap.set(15, -6)
    #cap.set(3, 1380)

    while cap.isOpened():
        t1 = cv2.getTickCount()
        _, frame = cap.read()
        lightGroup = ld.lightDetect(frame, armcolor)
        armorPixel = ad.armorDetect(frame, lightGroup)

        if showText:     #如果显示文本
            naf = pf.putMsg(frame, armorPixel, count) #打印信息
            fps.append(pf.putFps(frame, t1))
        if showimage:        #如果显示图片
            cv2.imshowimage("main", frame)
            if len(armorPixel) > 0:
                for x, y, w, h in armor:
                    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                    x, y, w, h = [i+1 for i in [x, y, w, h]]
                    image = frame[y: h, x: w]
                    cv2.imshowimage("armor", image)
                    #print(x,y,w,h)
        else:
            if naf > 400:       #远程操控妙算按键失效，自动退出
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
        pf.FpsTimeHist(fps)

if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    try:
        main(cam)
    except:
        main(cam)
