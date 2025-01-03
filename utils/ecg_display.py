"""
@Project ：
@File    ：ecg_display.py
@Author  ：DFQX
@Date    ：2022/9/18 13:09
@Description: 使用matplotlib打印心电图
"""

from matplotlib import pyplot as plt


'''
函数：plt_ecg
描述：用于绘制只有一条导联数据的心电图，并且时长不能太长
输入：data:心电数据；fs:采样率；gain:增益; path:保存图片路径，默认不保存
输出：一张心电图，没有保存为文件，如果需要，使用plt.savefig()
'''


def plt_ecg(data, fs=250, gain=1, path=''):  # 打印ECG图纸,一个心电图
    """
    用于绘制只有一条导联数据的心电图，并且时长不能太长
    :param data: list, 心电数据
    :param fs: int, 采样率
    :param gain: int, 增益
    :param path: str， 默认为空，不用保存，设置路径后保存图片
    :return: 无
    """
    x = [i / fs for i in range(0, len(data))]  # 计算x轴时间
    y = [val / gain for val in data]  # 根据增益计算幅值
    plt.figure(figsize=(len(data) * 25 // (fs * 14), 4))  # 画板大小固定，不要更改
    plt.xlabel("Time: s", fontsize=14)
    plt.ylabel("Voltage: mV", fontsize=14)
    plt.margins(x=0)
    ax = plt.gca()
    # 心电图纸：横坐标一般小格0.04s,大格0.2s; 纵坐标小格0.1mv，大格0.5mv
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))  # 设置x轴主刻度
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))  # 设置x轴次刻度
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))  # 设置y轴主刻度
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 设置y轴次刻度
    # 绘制大格和小格的线条
    ax.grid(which='major', axis="both", linewidth=0.75, linestyle='-', color='r')
    ax.grid(which='minor', axis="both", linewidth=0.25, linestyle='-', color='r')
    plt.ylim([-2.5, 2.5])  # y轴值为-2.5~2.5mV之间
    plt.plot(x, y, 'black', linewidth=0.9)  # 心电图形绘制
    plt.savefig(path) if path.strip() != '' else None
    plt.show()


def plot_peaks(data, locs, vals, title=''):
    """
    打印波峰位置
    :param data: list, 信号数据
    :param locs: list, 位置数据
    :param vals: list, peak的值
    :param title: str, 图形的标题, 默认为空
    :return: None
    """
    plt.figure(figsize=(13, 3))
    plt.plot(data)
    plt.plot(locs, vals, 'r*')
    plt.title(title)
    plt.show()


def subplot_peaks(data1, data2, locs, vals,title1='', title2=''):
    """
    打印信号2波峰位置, 同时将两条数据进行比对
    :param title2: str, 图形的标题, 默认为空
    :param title1: str, 图形的标题, 默认为空
    :param data1: list, 信号1数据
    :param data2: list, 信号2数据
    :param locs: list, 位置数据
    :param vals: list, peak的值
    :return: None
    """
    plt.figure(figsize=(13, 5))
    plt.subplot(211)
    plt.plot(data1)
    plt.title(title1)
    plt.subplot(212)
    plt.plot(data2)
    plt.plot(locs, vals, 'r*')
    plt.title(title2)
    plt.tight_layout()
    plt.show()


def plot_simple_comp(data1, data2, title1='', title2=''):
    """
    将两个信号进行简单的对比
    :param data1: 信号1
    :param data2: 信号2
    :param title1: 标题1
    :param title2: 标题2
    :return: None
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(211)
    plt.plot(data1)
    plt.title(title1)
    plt.subplot(212)
    plt.plot(data2)
    plt.title(title2)
    plt.tight_layout()
    plt.show()


def plot_peak_sig_and_noise_for_win_and_filter(data1, data2, x, y, x1, y1, locs1, th1, th2, th3,
                                               th4, title1='', title2=''):
    """
    绘制peak值的点
    :param data: 数据
    :param x: locs, 坐标位置
    :param y: peak值
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.title(title1)
    plt.plot(data1)
    plt.plot(x, y, '*r')
    plt.plot(locs1, th1)
    plt.plot(locs1, th2)
    plt.subplot(212)
    plt.title(title2)
    plt.plot(data2)
    plt.plot(x1, y1, 'og')
    plt.plot(locs1, th3)
    plt.plot(locs1, th4)
    plt.tight_layout()
    plt.show()


def plot_peak_dot(data, x, y):
    """
    绘制peak值的点,以及两个阈值的点
    :param data: 数据
    :param x: locs, 坐标位置
    :param y: peak值
    :return: None
    """
    plt.figure(figsize=(12, 4))
    plt.plot(data)
    plt.plot(x, y, '*r')
    plt.show()

def plot_peak_dot_th1_th2(data, x, y, th1, th2):
    """
    绘制peak值的点,以及两个阈值的点
    :param data: 数据
    :param x: locs, 坐标位置
    :param y: peak值
    :return: None
    """
    plt.figure(figsize=(12, 4))
    plt.plot(data)
    plt.plot(x, y, '*r')
    plt.plot(x, th1)
    plt.plot(x, th2)
    plt.show()


def plot_peak_dot_llv_rlv(data, x, y, LLV, RLV, th1, th2):
    """
    绘制peak值的点,以及两个阈值的点
    :param data: 数据
    :param x: locs, 坐标位置
    :param y: peak值
    :return: None
    """
    plt.figure(figsize=(12, 4))
    plt.plot(data)
    plt.plot(x, y, '*r')
    plt.plot(LLV)
    plt.plot(RLV)
    plt.plot(x, th1)
    plt.plot(x, th2)
    plt.show()


def plot_ecg_delay(x, y):
    """
    绘制ecg的延迟图
    """
    plt.figure(figsize=(5, 4))
    plt.plot(x, y)
    plt.show()


def plot_ecg_delay_3d(x, y, z):
    """
    绘制ecg的延迟图
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(5, 4))
    ax = Axes3D(fig)
    ax.plot(x, y, z)
    ax.tick_params(axis='x', colors='g')
    ax.tick_params(axis='y', colors='g')
    ax.tick_params(axis='z', colors='g')
    plt.show()


