## /usr/bin/python3
#coding=utf-8
# Official Package
import time
import threading
import cv2
import os

# Private Function Package
import armorDetect as ad
import lightDetect as ld
import pefermance as pf
import KalmanPredict as kp
import SerialSend as ss
import attackjudge as aj

# global variable
frame_x = 1280  # 全视窗模式分辨率
frame_y = 720
addx = 20
addy = 120
midx = 320      # 非全视窗模式分辨率
midy = 240
ad.debug_mode = True
serial_give = True
read_video = False       # 检测视频而不是检测摄像头
write_video = True      # 开启保存视频
ser = 0
key = 0

# camera load
out = None              # 保存视频流初始化
if not read_video:
    cap = cv2.VideoCapture(0)
    cap.set(3, frame_x)
    cap.set(4, frame_y)
    # cap.set(15, -8)      # 曝光度最低为-8
    EntireWindow = False # 全视窗模式
else:
    cap = cv2.VideoCapture('output.avi')
    EntireWindow = False  # 全视窗模式

# 配置灯条检测预处理参数
ld.frame_threshold = [50, 255]             # 二值化阈值
ld.aspect_threshold = [0.06, 0.6]           # 长宽比阈值
ld.red_down_threshold = [1, 1, 70]      # 红色阈值下界
ld.red_up_threshold = [180, 180, 255]       # 红色阈值上界
ld.blue_down_threshold = [130, 1, 1]   # 蓝色阈值下界
ld.blue_up_threshold = [255, 180, 180]      # 蓝色阈值上界

# 配置装甲检测参数
ad.length_threshold = 1.0                   # 灯条长度比
ad.width_threshold = 1.0                    # 灯条宽度比
ad.aspect_threshold = [0.9, 5.0]             # 长宽比
ad.ortho_threshold = [0.1, 0.1, 0.95]        # 正交率阈值(angle_l,angle_r,angle_p)
ad.target_num = None                        # 要检测的数字(如果为None表示不加入数字检测条件)
ad.ortho_mode = False						# 正交率划线显示
ad.bet_mode = False                         # 夹心灯条显示
ad.error_text = False						# 检测错误输出文本

# 动态选择数字攻击策略
aj.armor_len = 30      # buffer最大长度
aj.armor_correct = 8     # 最小选中数量

def stm32(debug_mode=False):
    global ser, key, midx, midy, serial_give
    while serial_give:
        try:
            serial_msg = ss.serial_send(ser, midx, midy)
            if debug_mode:
                print("You have sent", ord(serial_msg[0]), serial_msg[1], serial_msg[2], serial_msg[3], ord(serial_msg[4]), serial_msg[5], serial_msg[6], serial_msg[7], ord(serial_msg[-1]),"to the serial, x = ", midx, "y = ", midy)
            time.sleep(0.005)
            if key is ord('q'):
                return 
        except:
            try:
                ser = ss.serial_init(115200, 1)
                print("Here is no serial device, waiting for a connection...") 
                if key is ord('q') :
                    return
            except:
                pass

def main():
    """ 
    输入：cam(摄像头选取参数) 功能：主程序 输出：无 
    """
    global midx, midy, frame_x, frame_y
    global key, serial_give, cap
    color = 98  # 114: red, 98: blue
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        name = os.getcwd()
        filenum = os.listdir(name)
        filename = "{0}/{1}.avi".format(name, len(filenum))
        print(filename)
        if(os.path.exists(filename) == False):
            out = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
        else:
            print("error: find the same name video")
    else:
        out = None
    tiktok = pf.Tiktok()                    # 滴答计时器
    svm = cv2.ml.SVM_load('./svm_data.dat') # 读取svm参数
    kalman = kp.Kalman_Filter()             # 卡尔曼滤波器类初始化
    frame = pf.Frame(640, 480, frame_x, frame_y, addx, addy,
    EntireWindow, tiktok, focus = True, success=True, out = out
    )                                       # frame类

    while cap.isOpened():
        tiktok.tik()
        frame.update(cap)
        # choose "RGB" or "GRAY" to change pre-process raw image
        gray, group = ld.light_detect(frame, color, prepro = "RGB", preview = True) # preview为显示预处理图像
        # armorgroup = ad.armor_detect(svm, frame, group, train_mode=True, file="F:\\traindata\\"+str(target_num)) # 训练用，需要修改保存训练集目录
        armorgroup = ad.armor_detect(svm, frame.img, group, num_preview = True, train_mode=False)

        target, armor_list = aj.number_auto(armorgroup) # 动态数字选择
        print(target, armor_list)

        if armorgroup:
            if(target is not None):
                armor = aj.judge(armorgroup, attack = aj.mid, args = target)
                [midx, midy] = armor.mid
                armor.show(frame.frame_out, kalman, KalmanPreview = False)
                cv2.putText(frame.frame_out, str(armor.digit), tuple(armor.mid), cv2.FONT_ITALIC, 0.8,
                        (255, 255, 255), 2)
        else:
            midx = 320   # 未检测到的时候默认发串口为屏幕中心坐标
            midy = 240

        frame.imshow(armorgroup)

        _k, color = pf.key_detect(out, cap, color)
        if(_k):
            key = 113
            break

        tiktok.tok()

if __name__ == "__main__":
    if serial_give:   # if or if not serial
        ser = ss.serial_init(115200, 1)  # get a serial item, first arg is the boudrate, and the second is timeout
        if ser == 0:  # if not an existed Serial object
            print("Caution: Serial Not Found!")  # print caution
        thread_main = threading.Thread(target=main)
        thread_serial = threading.Thread(target=stm32, args=(serial_give,))
        thread_main.start()
        thread_serial.start()
        thread_main.join()
        thread_serial.join()
        print("Threads has been stopped")

    else: # None Serial Debug Connection
        main()
