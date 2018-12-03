#/usr/bin/python3
import cv2
import numpy as np
import sys
import math


def frameReady(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    readyDst = cv2.erode(cv2.dilate(thresh, kernel, iterations = 1), kernel, iterations = 1)
    return readyDst


def lightAspectDet(rectangle, contour):
    w, h = rectangle[1]
    if w == 0 or h == 0: return
    if w > h:  w, h = h, w
    return w/h > 0.05 and w/h < 0.5


def aimColormean(lightArea, mask):
    global mode
    mean_val = cv2.mean(lightArea, mask)
    if mode == ord("r"):
        meanVal = mean_val[2] > 220 and mean_val[2] > mean_val[0]
    if mode == ord("b"):
        meanVal = mean_val[0] > 220 and mean_val[0] > mean_val[2]
    return meanVal


def lightDetect(image):
    global mode
    lightGroup = []

    readyDst = frameReady(image)
    img, contours ,hierarchy = cv2.findContours(readyDst, 
            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        lightRectangle = cv2.minAreaRect(contour)
        x, y, w, h = cv2.boundingRect(contour)
        mask = readyDst[y: y+h, x: x+w]
        lightArea = image[y: y+h, x: x+w]
        lightArea = cv2.bitwise_and(lightArea, lightArea, mask = mask)
        if not (lightAspectDet(lightRectangle, contour) and aimColormean(lightArea, mask)):
            continue
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        lightGroup.append(lightRectangle)
    return lightGroup



def paralle(left, right):
    global frame
    [lx1, ly1], [lx2, ly2] = left[0], left[1]
    [rx1, ry1], [rx2, ry2] = right[3], right[2]
    cv2.line(frame, (lx1, ly1), (rx1, ry1), (255, 0, 0), 3)
    cv2.line(frame, (lx2, ly2), (rx2, ry2), (255, 0, 0), 3)
    lk = (ry2-ly1) / (rx2 - lx1)
    rk = (ry2-ly2) / (rx2 - lx2)
    print("lk: {0}\nrk: {1}".format(lk, rk))
    paralle = abs((lk - rk) / (1 + lk*rk))
    print("paralle :", paralle)
    return paralle < 7


def armorPixel(leftLight, rightLight):
    lpixel = cv2.boxPoints(leftLight)
    rpixel = cv2.boxPoints(rightLight)
    x = sorted(np.append(lpixel[0:4, 0], rpixel[0:4, 0]))
    y = sorted(np.append(lpixel[0:4, 1], rpixel[0:4, 1]))
    if paralle(lpixel, rpixel):
        return [x[0], y[0], x[7]-x[0], y[7]-y[0]]


def hightDifferenceDet(hLeft, hRight):
    hightDif = abs(hLeft-hRight) / max(hLeft, hRight)
    print("hightDif: ", hightDif)
    return  hightDif <= 0.15


def widthDifferenceDet(wLeft, wRight):
    widthDif = abs(wLeft-wRight) / max(wLeft, wRight)
    print("widthDif: ", widthDif)
    return widthDif <= 0.55


def armorAspectDet(xLeft, yLeft, xRight, yRight, hLeft, hRight, wLeft, wRight):
    armorAspect = math.sqrt((yRight-yLeft)**2 + (xRight-xLeft)**2) / max(hLeft, hRight, wLeft, wRight)
    print("armorAspect: ", armorAspect)
    return (7 >= armorAspect and armorAspect >= 6) or (3 >= armorAspect and armorAspect >= 2)


def isArmor(leftLight, rightLight):
    [xLeft, yLeft], [wLeft, hLeft] = leftLight[0], leftLight[1]
    [xRight, yRight], [wRight, hRight] = rightLight[0], rightLight[1]
    if wLeft > hLeft:
        wLeft, hLeft = hLeft, wLeft
    if wRight > hRight:
        wRight, hRight = hRight, wRight
    return (hightDifferenceDet(hLeft, hRight) and
            widthDifferenceDet(wLeft, wRight) and 
            armorAspectDet(xLeft, yLeft, xRight, yRight, hLeft, hRight, wLeft, wRight))


def armorDetect(lightGroup):
    armorArea = []
    for left in range(len(lightGroup)):
        for right in range(left + 1, len(lightGroup)):
            if lightGroup[left][0][0] > lightGroup[right][0][0]:
                left, right = right, left
            if not isArmor(lightGroup[left], lightGroup[right]):
                continue
            armor = armorPixel(lightGroup[left], lightGroup[right])
            if armor != None:
                armorArea.append(armor)
            print(armor)
    return armorArea


def measurement(frame, e1):
    time = (cv2.getTickCount() - e1) / cv2.getTickFrequency()
    fps = "FPS:{0:0.2f}".format(1/time)
    frame = cv2.putText(frame, fps, (50, 50), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
    cv2.imshow("frame", frame)


if __name__ == "__main__":

    try:
        cam = sys.argv[1]
    except:
        cam = 0

    cap = cv2.VideoCapture(cam)
    cap.set(15, -5)
    mode = ord("r")
    cv2.namedWindow("frame")
    
    while cap.isOpened:
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        e1 = cv2.getTickCount()
        _, frame = cap.read()
        armor = armorDetect(lightDetect(frame))
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        if len(armor) > 0:
            for x, y, w, h in armor:
                cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 0, 255), 2)
                print("success".center(50, "-"))
        measurement(frame, e1)
        key = cv2.waitKey(5)
        if key == ord("r") or key == ord("b"):
            mode = key
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
