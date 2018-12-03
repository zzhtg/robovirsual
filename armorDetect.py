import cv2
import numpy as np
import math


def paralle(left, right):
    [lx1, ly1], [lx2, ly2] = left[0], left[2]
    [rx1, ry1], [rx2, ry2] = right[0], right[2]
    lk = (ry1-ly1) / (rx1 - lx1)
    rk = (ry2-ly2) / (rx2 - lx2)
    paralle = abs((rk - lk) / (1 + lk*rk))
    return paralle < 0.05


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
    return (7.5 >= armorAspect and armorAspect >= 6) or (3.5 >= armorAspect and armorAspect >= 2)


def isArmor(leftLight, rightLight):
    [xLeft, yLeft], [wLeft, hLeft] = leftLight[0], leftLight[1]
    [xRight, yRight], [wRight, hRight] = rightLight[0], rightLight[1]
    if wLeft > hLeft: wLeft, hLeft = hLeft, wLeft
    if wRight > hRight: wRight, hRight = hRight, wRight
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
                #print(armor)
    return armorArea


