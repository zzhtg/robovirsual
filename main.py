#/usr/bin/python3
import cv2
import numpy as np
import sys
import math
from lightDetect import *
from armorDetect import *


def measurement(frame, e1):
    time = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    fps = "FPS:{0:0.2f}".format(1/time)
    frame = cv2.putText(frame, fps, (50, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    cv2.imshow("frame", frame)


if __name__ == "__main__":

    listCount = []
    listFcount = []
    success = 0

    try:
        cam = sys.argv[1]
    except:
        cam = 0

    cap = cv2.VideoCapture(cam)
    cap.set(3, 1390)
    cap.set(15, -5)
    mode = ord("r")
    cv2.namedWindow("frame")
    
    while cap.isOpened:
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        _, frame = cap.read()
        e1 = cv2.getTickCount()
        armor = armorDetect(lightDetect(frame, mode))
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        if len(armor) > 0:
            x, y, w, h = armor[0]
            cv2.rectangle(frame, (int(x), int(y-(h/2))), (int(x)+int(w), int(y)+int(h)), (0, 0, 255), 2)
        measurement(frame, e1)
        key = cv2.waitKey(5)
        if key == ord("r") or key == ord("b"):
            mode = key
            print("{}".format(mode))
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
