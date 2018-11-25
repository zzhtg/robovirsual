import cv2
import numpy as np

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


def aimColormean(lightArea, mask, mode):
    mean_val = cv2.mean(lightArea, mask)
    if mode == ord("r"):
        meanVal = mean_val[2] > 220 and mean_val[2] > mean_val[0]
    if mode == ord("b"):
        meanVal = mean_val[0] > 220 and mean_val[0] > mean_val[2]
    return meanVal


def lightDetect(image, mode):
    """
    对于colorDetect()函数颜色选择后的mask进行Canny等处理
    寻找边界、选择符合条件的边界
    返回符合条件的灯条列表以及截取区域的坐标
    """
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
        if not (lightAspectDet(lightRectangle) and aimColormean(lightArea, mask, mode)):
            continue
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        lightGroup.append(lightRectangle)
    return lightGroup
