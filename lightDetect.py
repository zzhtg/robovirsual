# coding=utf-8
import cv2
import numpy as np
frame_threshold = [150, 255]
aspect_threshold = [0.06, 0.5]
red_down_threshold = [60, 110, 220]
red_up_threshold = [180, 220, 255]
blue_down_threshold = [220, 150, 30]
blue_up_threshold = [255, 250, 230]
class Light():
    def __init__(self, rect, color, aspect):
        self.color = color
        self.aspect = aspect
        self.rect = [rect[0][0], rect[0][1], rect[1][0], rect[1][1]]
        self.raw = rect
        if(self.rect[2] < self.rect[3]):
            self.rect[2], self.rect[3] = self.rect[3], self.rect[2]

def frame_ready(image, detect = "RGB", color = 98, preview = False):
    """
    输入：image(当前帧图像)
    功能：灰度、二值化、膨胀再腐蚀等处理
    输出：image（处理后图像）
    """
    # traditional gray threshold
    def get_gray(image):
        gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        down, up = frame_threshold
        return cv2.threshold(gray, down, up, cv2.THRESH_BINARY)[1]

    # inrange BGR threshold in three channel
    def get_rgb(image):
        rgb_down = np.array((0, 0, 0), dtype="uint8")
        rgb_up = np.array((255, 255, 255), dtype="uint8")

        if color is 114:  # red
            rgb_down = np.array(red_down_threshold, dtype="uint8")
            rgb_up = np.array(red_up_threshold, dtype="uint8")

        elif color is 98:  # blue
            rgb_down = np.array(blue_down_threshold, dtype="uint8")
            rgb_up = np.array(blue_up_threshold, dtype="uint8")

        return cv2.inRange(image, rgb_down, rgb_up)

    if(detect is "RGB"):
        image = get_rgb(image)
    elif(detect is "GRAY"):
        image = get_gray(image)

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(cv2.dilate(image, kernel, iterations=1), kernel, iterations=1)
    if(preview):
        cv2.imshow("ready", image)
    return image

def detect(frame, pretreatment, contour, color):
    def light_aspect_det():
        """
        输入：rectangle(灯条最小拟合矩形数据)
	    功能：检测灯条边界矩形的横纵比是否符合条件
	    输出：True 或者 False、aspect（灯条横纵比）
        """
        w, h = rect[1]
        if w > h:
            w, h = h, w
        aspect = (w + 1) / (h + 1)
        return aspect_threshold[0] < aspect <aspect_threshold[1], aspect

    def aim_color_mean():
        """
        输入：area(灯条最小拟合矩形图像)、mask(灯条发光区域)、color(装甲颜色)
	    功能：检测灯条颜色是否符合条件（BGR颜色空间当前红色或蓝色分量是否够大）
	    输出：_c(True 或者 False)、value(颜色空间平均值)
	    """
        value = cv2.mean(area, mask)
        _c = False
        if color is 114: # red
            _c = [(red_down_threshold[i] <= value[i] <= red_up_threshold[i]) for i in range(0, 3)]
            return _c, value
        if color is 98: # blue
            _c = [(blue_down_threshold[i] <= value[i] <= blue_up_threshold[i]) for i in range(0, 3)]
            return _c, value

    font = cv2.FONT_ITALIC
    rect = cv2.minAreaRect(contour)

    l_, aspect = light_aspect_det()
    if not l_:
        msg = "aspect{0:.1f}-{1:.1f}-{2:.1f}".format(aspect, rect[1][0], rect[1][1])
        # cv2.putText(image, msg, (int(x), int(y)), font, 0.4, (0, 255, 0), 1)
        return False

    x, y, w, h = cv2.boundingRect(contour)
    area = frame.img[y: y+h, x: x+w]
    mask = pretreatment[y: y+h, x: x+w]
    area = cv2.bitwise_and(area, area, mask=mask)
    c_, value = aim_color_mean()

    if False in c_:
        msg = "value{0:.0f},{1:.0f},{2:.0f}".format(value[0], value[1], value[2])
        cv2.putText(frame.frame_out, msg, (x, y-15), font, 0.4, (0, 255, 0), 1)
        return False
    return rect, value, aspect

def light_detect(frame, color, prepro = "RGB", preview = False):
    """
    输入：image(当前帧图像)、color(装甲颜色)
    功能：找到符合颜色、横纵比条件的类似灯条的矩形
    输出：符合条件的矩形（最小边界拟合信息）的列表，不一定就是灯条
    """
    group = []
    pretreatment = frame_ready(frame.img, prepro, color, preview)
    # cv2 4.0.0 finContours返回轮廓和层级
    # cv2 3 finContours 返回图像、轮廓和层级
    # 修改1 为 0即可
    version = cv2.__version__
    v = 0 if (version[0] == '4') else 1 
    for contour in cv2.findContours(pretreatment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[v]:
        ans = detect(frame, pretreatment, contour, color)
        if(ans == False):
            continue
        else:
            rect, value, aspect = ans
        L = Light(rect, value, aspect) # 创建灯条类对象
        group.append(L)  # 添加至列表
    return pretreatment, group
