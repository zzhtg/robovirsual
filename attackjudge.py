import armorDetect as ad
import math
def func_mid(armorgroup, args = None):                  # 选择离中心最近的装甲
    ad.target_num = args                                # 关闭数字检测
    def center(mid):                                    # 装甲中心到准心的距离
        return math.sqrt((mid[0]-320)**2 + (mid[1]-240)**2)
    mid = [center(armor.mid) for armor in armorgroup]
    return armorgroup[mid.index(min(mid))]

def func_near(armorgroup, args = None):                 # 选择距离最近的装甲
    ad.target_num = args                                # 关闭数字检测
    dist_group = [armor.dist for armor in armorgroup]   # 装甲测距
    return armorgroup[dist_group.index(min(dist_group))]
# 以下两个功能得确定数字识别率才能使用
def func_number_static(armorgroup, args):               # 静态选择固定数字
    pass

def func_number_auto(armorgroup, args):                       # 动态选择固定数字
    pass
mid = 'ATTACK_MID'
near = 'ATTACK_NEAR'
static = 'ATTACK_NUMBER_STATIC'
auto = 'ATTACK_NUMBER_AUTO'
attack_mode = {               # 攻击策略字典
    mid: func_mid,
    near: func_near,
    static: func_number_static,
    auto: func_number_auto
}
def judge(armorgroup, attack = mid, args = None):
    mode = attack_mode.get(attack) # 从字典中获取攻击策略函数
    return mode(armorgroup, args)
