# coding=utf-8
import cv2
import time
import KalmanPredict as kp
count = {"perSucRatio": 0, "entire_success_ratio": 0, "alFrame": 0,
         "alSuc": 0, "perFrame": 0, "perSuc": 0, "period": 30}

class Tiktok():
    def __init__(self):
        self.start = 0.0
        self.end = 0.0
        self.interval = 1.0
    def print(self):
        print("FPS = ", self.interval)
    def tik(self):
        self.start = time.clock()
    def tok(self):
        self.end = time.clock()
        self.interval = (self.end - self.start)
    def put_success(self, frame, armorflag):
        """
        输入：frame(当前帧)、 armor(装甲列表)、count(计数成员字典)
        功能：当前帧添加实时成功率与全过程成功率、画出装甲图像
        输出：无
        """
        count["alFrame"] += 1
        count["perFrame"] += 1
        count["entire_success_ratio"] = count["alSuc"] / count["alFrame"]
        if armorflag:
            count["alSuc"] += 1
            count["perSuc"] += 1
        if count["alFrame"] % count["period"] is 0:
            count["perSucRatio"] = count["perSuc"] / count["perFrame"]
            count["perFrame"] = count["perSuc"] = 0

        font = cv2.FONT_ITALIC
        fps = 1.0 / self.interval
        msg = "fps:{0:0.2f} real_time:{1:0.2f}  entire_time:{2:0.2f}".format(
            fps, count["perSucRatio"], count["entire_success_ratio"])
        cv2.putText(frame, msg, (50, 50), font, 0.8, (255, 255, 255), 2)

class Frame():
    def __init__(self, ratex, ratey, framex, framey, EntireWindow,
                 tiktok, focus = True, success = True, out = None):
        self.entireflag = EntireWindow
        self.ratex = ratex # 缩放窗口大小 640 x 480
        self.ratey = ratey
        self.framex = framex
        self.framey = framey
        self.focus = focus
        self.success = success
        self.tiktok = tiktok
        self.out = out
        self.frame_out = None

    def update(self, cap):
        _, img = cap.read()
        if img is not None:
            if(self.entireflag):
                self.img = img # 完整模式
            else:
                x1 = int(self.framey / 2 - self.ratey / 2)
                x2 = int(self.framey / 2 + self.ratey / 2)
                y1 = int(self.framex / 2 - self.ratex / 2)
                y2 = int(self.framex / 2 + self.ratex / 2)
                self.img = img[x1: x2, y1: y2]   # 缩放模式
            self.frame_out = self.img.copy()# 将调试信息放在拷贝里面

    def imshow(self, armorflag):
        def fun_focus():  # 画出十字线和准星
            x = int(self.ratex / 2.0)
            y = int(self.ratey / 2.0)
            cv2.line(self.frame_out, (x, 0), (x, 2 * y - 1), (255, 255, 255), 3)
            cv2.line(self.frame_out, (0, y), (2 * x - 1, y), (255, 255, 255), 3)
        if(self.focus):
            fun_focus()
        if(self.success):
            self.tiktok.put_success(self.frame_out, armorflag)
        cv2.imshow("frame", self.frame_out)
        if(self.out is not None):
            self.out.write(self.img)

def key_detect(out, cap, color, delay = 5):
    key = cv2.waitKey(delay)
    if key is ord('r') or key is ord('b'):
        color = key
    if key is ord('q'):
        if(out is not None):
            out.release()
        cap.release()  # 摄像头关闭
        cv2.destroyAllWindows()
        return True, color
    return False, color

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


def show_kalman(frame, pos, matrix, kalman, error, real_error, preview = False):
    if pos:
        x, y, w, h = pos
        matrix, error, real_error = kp.predict(matrix, kalman, error, real_error, frame, x, y, w, h, preview)

