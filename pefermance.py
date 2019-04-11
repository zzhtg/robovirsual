# coding=utf-8

import cv2
import numpy as np
import KalmanPredict as kp


def put_cross_focus(frame):
    """
    :param frame: 输入帧图像
    :return: 画图后的图像
    :function: 画出十字线和准星
    """
    cv2.line(frame, (320, 0), (320, 479), (255, 255, 255), 3)
    cv2.line(frame, (0, 240), (639, 240), (255, 255, 255), 3)
    return frame


count = {"perSucRatio": 0, "entire_success_ratio": 0, "alFrame": 0,
         "alSuc": 0, "perFrame": 0, "perSuc": 0, "period": 30}


def put_success(frame, interval, armor, count):
    """
    输入：frame(当前帧)、 armor(装甲列表)、count(计数成员字典)
    功能：当前帧添加实时成功率与全过程成功率、画出装甲图像
    输出：无
    """
    count["alFrame"] += 1
    count["perFrame"] += 1
    count["entire_success_ratio"] = count["alSuc"] / count["alFrame"]
    if armor:
        count["alSuc"] += 1
        count["perSuc"] += 1
    if count["alFrame"] % count["period"] is 0:
        count["perSucRatio"] = count["perSuc"] / count["perFrame"]
        count["perFrame"] = count["perSuc"] = 0
    font = cv2.FONT_ITALIC
    fps = 1.0 / interval
    msg = "fps:{0:0.2f} real_time:{1:0.2f}  entire_time:{2:0.2f}".format(
                fps, count["perSucRatio"], count["entire_success_ratio"])
    cv2.putText(frame, msg, (50, 50), font, 0.8, (255, 255, 255), 2)
    return count["alFrame"]


def fps_count(fps):
    """
    输入：fps(包含fps信息的列表)
    功能：画出每帧对应的fps，显示执行过程当中最大、最小和平均帧率
    输出：无
    """
    fps.sort()
    fps[0:] = fps[1:]
    sum_fps = 0
    for num in fps:
        sum_fps += num
    print("average fps = {0:0.1f}".format(sum_fps / len(fps)))
    print("max fps = {0:0.1f}".format(fps[len(fps) - 1]))
    print("min fps = {0:0.1f}".format(fps[0]))


def show_kalman(frame, coordinate, matrix, kalman, error, real_error):
    if coordinate:
        x, y, w, h = coordinate[0][0:4]
        matrix, error, real_error = kp.predict(matrix, kalman, error, real_error, frame, x, y, w, h)

