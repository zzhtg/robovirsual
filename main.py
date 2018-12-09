#/usr/bin/python3
import cv2
import numpy as np
import sys
from armorDetect import *
from lightDetect import *


def measurement(frame, e1):
    time = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    fps = "FPS:{0:0.2f}".format(1/time)
    frame = cv2.putText(frame, fps, (50, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    cv2.imshow("frame", frame)


def main(cam):
    nsu = asu = naf = nar = nnf = nnr = 0
    t = 20

    cap = cv2.VideoCapture(cam)
    cap.set(15, -5)
    mode = 114  #114: red, 98: blue
    cv2.namedWindow("frame")

    while 1:
        naf += 1
        nnf += 1
        asu = nar / naf
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        e1 = cv2.getTickCount()
        _, frame = cap.read()
        armor = armorDetect(lightDetect(frame, mode))
        measurement(frame, e1)
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        if len(armor) > 0:
            for x, y, w, h in armor:
                #cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 0, 255), 2)
                image = frame[y:y+h, x:x+w]
                image = cv2.resize(image, (300, 100))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cv2.imshow("image", image)
                #print(x,y,w,h)
            nar += 1
            nnr += 1
        if naf%t is 0:
            nsu = nnr / nnf
            nnf = nnr = 0
        cv2.putText(frame, "intime:{0:0.2f}  alltime:{1:0.2f}".format(nsu, asu
            ), (250, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        key = cv2.waitKey(1)
        if key is 114 or key is 98:
            mode = key
        if key is ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":

    try:
        cam = sys.argv[1]
    except:
        cam = 0
    main(cam)
