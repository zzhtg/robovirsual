import cv2
import numpy as np


def frameReady(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    readyDst = cv2.erode(cv2.dilate(thresh, kernel, iterations = 1), kernel, iterations = 1)
    return readyDst


def lightAspectDet(rectangle):
    w, h = rectangle[1]
    if w == 0 or h == 0: return
    if w > h:  w, h = h, w
    return w/h > 0.1 and w/h < 0.4


def aimColormean(lightArea, mask, mode):
    mean_val = cv2.mean(lightArea, mask)
    #print("mean_val: ", mean_val)
    if mode == 114:
        meanVal = mean_val[2] > 200 and mean_val[2] > mean_val[0]
    if mode == 98:
        meanVal = mean_val[0] > 200 and mean_val[0] > mean_val[2]
    return meanVal


def lightDetect(image, mode):
    lightGroup = []
    readyDst = frameReady(image)
    for contour in cv2.findContours(readyDst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]:
        lightRectangle = cv2.minAreaRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if not lightAspectDet(lightRectangle):
            continue
        mask = readyDst[y: y+h, x: x+w]
        lightArea = image[y: y+h, x: x+w]
        lightArea = cv2.bitwise_and(lightArea, lightArea, mask = mask)
        if not aimColormean(lightArea, mask, mode):
            continue
        #cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        lightGroup.append(lightRectangle)
    return lightGroup


