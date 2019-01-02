#/usr/bin/python3
import cv2
import numpy as np
import sys
from armorDetect import *
from lightDetect import *
from measurement import *


def main(cam):
    cap = cv2.VideoCapture(cam)
    cap.set(15, -5)
    mode = 114  #114: red, 98: blue
    cv2.namedWindow("frame")
    ns = [0, 0, 0, 0, 0, 0, 20]

    while cap.isOpened():
        e1 = cv2.getTickCount()
        _, frame = cap.read()
        armor = armorDetect(lightDetect(frame, mode))

        naf = putMsg(frame, armor, ns)
        measurement(frame, e1)

        key = cv2.waitKey(1)
        if key is 114 or key is 98:
            mode = key
        if key is ord('q'):
            break
        if naf >= 100:
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    try:
        cam = sys.argv[1]
    except:
        cam = 0

    main(cam)
