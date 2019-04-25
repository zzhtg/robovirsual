import cv2
import numpy as np
# 其实下面四个变量就是原来的matrix
def kalman_init():
    last_mes = current_mes = np.array((2, 1), np.float32)
    last_pre = current_pre = np.array((2, 1), np.float32)
    kalman = cv2.KalmanFilter(4, 2, 0)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
    return kalman
class Kalman_Filter():
    def __init__(self):
        self.last_mes = [0, 0]
        self.current_mes = [0, 0]
        self.last_pre = [0, 0]
        self.current_pre = [0, 0]
        self.error = []
        self.real_error = []
        self.kalman = kalman_init()
    def predict(self, frame, pos, preview = False):
        x1, y1, x2, y2 = pos
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        self.current_mes = np.array([[np.float32(x)], [np.float32(y)]])  # 现在测量的位置
        self.last_pre = self.current_pre  # 更新预测值
        self.last_mes = self.current_mes  # 更新测量值

        self.kalman.correct(self.current_mes)  # 用实际值纠正测量值
        self.current_pre = self.kalman.predict()  # 预测

        lmx, lmy = self.last_mes[0], self.last_mes[1]  # last measure
        lpx, lpy = self.last_pre[0], self.last_pre[1]  # last predict
        cmx, cmy = self.current_mes[0], self.current_mes[1]  # current measure
        cpx, cpy = self.current_pre[0], self.current_pre[1]  # current predict

        self.error.append(np.sqrt(cpx * cpx + cpy * cpy))
        self.real_error.append(np.sqrt(cmx * cmx + cmy * cmy))
        cv2.rectangle(frame, (cmx - (x2 - x1) / 2, cmy - (y2 - y1) / 2), (cmx + (x2 - x1) / 2, cmy + (y2 - y1) / 2),
                      (0, 0, 255), 3)
        # if(preview):
        #     cv2.rectangle(frame, (cpx-(x2-x1)/2, cpy-(y2-y1)/2), (cpx+(x2-x1)/2, cpy+(y2-y1)/2), (0, 255, 0), 3)
