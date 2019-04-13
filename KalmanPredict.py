import cv2
import numpy as np
error = []
real_error = []


def kalman_init():
    last_mes = current_mes = np.array((2, 1), np.float32)
    last_pre = current_pre = np.array((2, 1), np.float32)
    matrix = [last_mes, current_mes, last_pre, current_pre]
    kalman = cv2.KalmanFilter(4, 2, 0)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
    return matrix, kalman


def predict(matrix, kalman, error, real_error, frame, x1, y1, x2, y2):
    last_mes, current_mes, last_pre, current_pre = matrix
    last_pre = current_pre  # 更新预测值
    last_mes = current_mes  # 更新测量值
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    current_mes = np.array([[np.float32(x)], [np.float32(y)]])  # 现在测量的位置
    kalman.correct(current_mes)  # 用实际值纠正测量值
    current_pre = kalman.predict()  # 预测

    lmx, lmy = last_mes[0], last_mes[1]  # last measure
    lpx, lpy = last_pre[0], last_pre[1]  # last predict
    cmx, cmy = current_mes[0], current_mes[1]  # current measure
    cpx, cpy = current_pre[0], current_pre[1]  # current predict

    error.append(np.sqrt(cpx * cpx + cpy * cpy))
    real_error.append(np.sqrt(cmx * cmx + cmy * cmy))
    cv2.rectangle(frame, (cmx - (x2 - x1) / 2, cmy - (y2 - y1) / 2), (cmx + (x2 - x1) / 2, cmy + (y2 - y1) / 2),
                  (0, 0, 255), 3)
    cv2.rectangle(frame, (cpx-(x2-x1)/2, cpy-(y2-y1)/2), (cpx+(x2-x1)/2, cpy+(y2-y1)/2), (0, 255, 0), 3)
    matrix = [last_mes, current_mes, last_pre, current_pre]
    return matrix, error, real_error
