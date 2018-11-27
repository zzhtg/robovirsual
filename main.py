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
    cv2.imshow("ready", readyDst)
    return readyDst


def lightAspectDet(rectangle):
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
    """
    对于colorDetect()函数颜色选择后的mask进行Canny等处理
    寻找边界、选择符合条件的边界
    返回符合条件的灯条列表以及截取区域的坐标
    """
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
        if not (lightAspectDet(lightRectangle) and aimColormean(lightArea, mask)):
            continue
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        lightGroup.append(lightRectangle)
    return lightGroup


def parallelDet(aLeft, aRight):
    paralle = abs(abs(aLeft + 45) - abs(aRight + 45))
    print("----------------------\nparalle: {0}".format(paralle))
    return paralle < 7


def hightDifferenceDet(hLeft, hRight):
    hightDif = abs(hLeft-hRight) / max(hLeft, hRight)
    print("hightDif: {0}".format(hightDif))
    return  hightDif <= 0.9


def widthDifferenceDet(wLeft, wRight):
    widthDif = abs(wLeft-wRight) / max(wLeft, wRight)
    print("widthDif: {0}".format(widthDif))
    return widthDif <= 0.9


def armorAspectDet(xLeft, yLeft, xRight, yRight, hLeft, hRight, wLeft, wRight):
    armorAspect = math.sqrt((yRight-yLeft)**2 + (xRight-xLeft)**2) / max(hLeft, hRight, wLeft, wRight)
    print("aspectDet: {0}".format(armorAspect))
    return (7 >= armorAspect and armorAspect >= 5) or (3 >= armorAspect and armorAspect >= 2)


def isArmor(leftLight, rightLight):
    [xLeft, yLeft], [wLeft, hLeft], aLeft = leftLight[0], leftLight[1], leftLight[2]
    [xRight, yRight], [wRight, hRight], aRight = rightLight[0], rightLight[1], rightLight[2]
    return (parallelDet(aLeft, aRight) and 
            hightDifferenceDet(hLeft, hRight) and
            widthDifferenceDet(wLeft, wRight) and 
            armorAspectDet(xLeft, yLeft, xRight, yRight, hLeft, hRight, wLeft, wRight))


def armorDetect(lightGroup):
    """
    对于lightDetect()函数返回的灯条列表
    找到符合条件的灯条组合
    """

    armorArea = []
    for left in range(len(lightGroup)):
        for right in range(left + 1, len(lightGroup)):
            if lightGroup[left][0][0] > lightGroup[right][0][0]:
                left, right = right, left
            if not isArmor(lightGroup[left], lightGroup[right]):
                continue
            print("succesee".center(50, ">"))

            lpixel = cv2.boxPoints(lightGroup[left])
            rpixel = cv2.boxPoints(lightGroup[right])
            xminp = sorted(np.append(lpixel[0:4, 0], rpixel[0:4, 0]))[0]
            xmaxp = sorted(np.append(lpixel[0:4, 0], rpixel[0:4, 0]))[7]
            yminp = sorted(np.append(lpixel[0:4, 1], rpixel[0:4, 1]))[0]
            ymaxp = sorted(np.append(lpixel[0:4, 1], rpixel[0:4, 1]))[7]

            armor = [xminp, yminp, xmaxp-xminp, ymaxp-yminp]
            armorArea.append(armor)
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
    cap.set(3, 1390)
    cap.set(15, -5)
    mode = ord("r")
    cv2.namedWindow("frame")
    
    while cap.isOpened:
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        _, frame = cap.read()
        e1 = cv2.getTickCount()
        armor = armorDetect(lightDetect(frame))
#---------^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^-------------------------------
        if len(armor) > 0:
            x, y, w, h = armor[0]
            cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 0, 255), 2)
        measurement(frame, e1)
        key = cv2.waitKey(5)
        if key == ord("r") or key == ord("b"):
            mode = key
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
