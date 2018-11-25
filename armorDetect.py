import cv2
import numpy as np
import math


def parallelDet(aLeft, aRight):
    paralle = abs(abs(aLeft + 45) - abs(aRight + 45))
    #print("----------------------\nparalle: {0}".format(paralle))
    return paralle < 25


def hightDifferenceDet(hLeft, hRight):
    hightDif = hLeft+hRight / max(hLeft, hRight)
    #print("hightDif: {0}".format(hightDif))
    return  hightDif >= 1.5


def widthDifferenceDet(wLeft, wRight):
    widthDif = wLeft+wRight / max(wLeft, wRight)
    #print("widthDif: {0}".format(widthDif))
    return widthDif >= 1.5


def armorAspectDet(xLeft, yLeft, xRight, yRight, hLeft, hRight, wLeft, wRight):
    armorAspect = math.sqrt((yRight-yLeft)**2 + (xRight-xLeft)**2) / max(hLeft, hRight, wLeft, wRight)
    #print("aspectDet: {0}".format(armorAspect))
    return 7 >= armorAspect and armorAspect >= 2
            


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
            xArmor = lightGroup[left][0][0]
            yArmor = lightGroup[right][0][1]
            wArmor = lightGroup[right][0][0] - lightGroup[left][0][0]
            hArmor = max(lightGroup[right][1][1], lightGroup[left][1][1])
            armor = [xArmor, yArmor, wArmor, hArmor]
            armorArea.append(armor)
    return armorArea
