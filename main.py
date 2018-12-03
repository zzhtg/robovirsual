#/usr/bin/python3
import cv2
import numpy as np
import sys
from armorDetect import *
from lightDetect import *
from parameter import * 


def measurement(frame, e1):
    time = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    fps = "FPS:{0:0.2f}".format(1/time)
    frame = cv2.putText(frame, fps, (50, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    cv2.imshow("frame", frame)


def main(cam):
    cap = cv2.VideoCapture(cam)
    cap.set(15, -4)
    mode = ord("r")
    cv2.namedWindow("frame")
    while cap.isOpened:
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        e1 = cv2.getTickCount()
        _, frame = cap.read()
        armor = armorDetect(lightDetect(frame, mode))
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        if len(armor) > 0:
            for x, y, w, h in armor:
                cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 0, 255), 2)
        measurement(frame, e1)
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        key = cv2.waitKey(5)
        if key == ord("r") or key == ord("b"):
            mode = key
        if key == ord('q'):
            break


if __name__ == "__main__":

    try:
        cam = sys.argv[1]
    except:
        cam = 0
    main(cam)
    cv2.destroyAllWindows()
    cap.release()
