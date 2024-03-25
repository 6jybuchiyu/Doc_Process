import os

import mne
import numpy as np
import torch
from matplotlib import pyplot as plt


# 画功率谱密度图
def plot_psd(raw, PSD_folder):
    bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 50]}
    for band in bands:
        psd_fig = raw.compute_psd(fmin=bands[band][0], fmax=bands[band][1], show=False).plot()
        psd_fig.savefig(os.path.join(PSD_folder, "psd_" + bands[band][0] + '-' + bands[band][1] + 'Hz' + ".jpg"))


def show_EEG(raw, scaling):
    # 显示全时间段和全通道的EEG数据，缩放因子要做一定调整
    raw.plot(duration=raw.n_times / raw.info['sfreq'], scalings={'eeg': scaling}, n_channels=raw.info['nchan'])
    plt.show()


# get所有被试下的CNT文件的路径
def get_CNT_path(data_folder):
    cnt_paths = []
    subject_folders = os.listdir(os.path.join(data_folder, 'raw'))
    subject_folders.sort()
    for subject_folder in subject_folders:
        origin_data_folder = os.path.join(data_folder, 'raw', subject_folder, 'Acquisition')
        cnt_paths.append(os.path.join(origin_data_folder, 'Acquisition 01_proc_convert.cdt.cnt'))
    return cnt_paths


def read_data(cnt_path):
    raw = mne.io.read_raw_cnt(cnt_path, preload=True)
    show_EEG(raw, 3e-3)
    # ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4',
    # 'T4','TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'Pz', 'P4', 'T6', 'A2', 'O1', 'Oz', 'O2']
    # 删除无用通道
    raw.drop_channels(['HEOL', 'VEOU', 'HEOR-L', 'VEOL-U', 'Trigger', 'A1', 'A2', 'TP7', 'TP8'])

    # 电极定位
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # 陷波滤波
    raw = raw.notch_filter(freqs=50)
    raw = raw.notch_filter(freqs=60)
    # ⾼通滤波为了消除电压漂移，低通滤波为了消除⾼频噪⾳
    raw = raw.filter(l_freq=0.1, h_freq=50)

    # 重参考
    """
        # 默认小放大器已经是用Fz参考
        raw.set_eeg_reference(ref_channels=['Fz'])
        # 比如采样A1 A2两个乳突电极进行重参考
        raw.set_eeg_reference(ref_channels=["A1", "A2"])  # 基于A1 A2完成重参考
        # 或者平均参考
        raw.set_eeg_reference(ref_channels='average')
        raw.plot(start=20, duration=1, n_channels=64, block=True, title='重参考完成，无误请关闭窗口')
    """
    return raw


def split_data(raw, subject_name, DataFolder):
    def crop_cnt_data(raw_copy, start, duration_second):
        raw_copy.load_data()
        start_second = start / raw_copy.info['sfreq']
        end_second = start_second + duration_second - 1.0 / raw_copy.info['sfreq']
        # 获取指定tmin秒到tmax秒位置的eeg片段
        return raw_copy.crop(tmin=start_second, tmax=end_second)

    def save_data4pt(cnt_data, DataFolder, name, trial):
        data, _ = cnt_data[:, :]
        tensor_file = torch.Tensor(data)
        path_save = os.path.join(DataFolder, "EEG", name)
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        save_file = trial + ".pt"
        torch.save(tensor_file, path_save + '\\' + save_file)
        print(torch.load(path_save + '\\' + save_file).shape)  # torch.Size([28, 30000])

    # 找到并裁剪出我们需要的片段，比如说标记了的事件相关信号
    events = mne.events_from_annotations(raw)  # 将raw里面的annotation读取出来
    eventTimeArr = events[0][:, 0]  # 0索引中所有行第一列的数值 打标点
    eventMarkArr = events[0][:, 2]  # 0索引中所有行第三列的数值 标记值
    roundStartMark = events[1]['80']  # 1索引中关键字为80的值 round开始的标记值
    roundEndMark = events[1]['88']  # 1索引中关键字为88的值 round结束的标记值
    roundStartMarkArr = np.where(eventMarkArr == roundStartMark)[0]  # 返回是(condition, x, y)
    roundEndMarkArr = np.where(eventMarkArr == roundEndMark)[0]  # condition这个元组代表实验开始/结束标记在events里出现的位置
    roundNumber = len(roundEndMarkArr)  # 结束标志的次数即该被试的实验轮数

    for roundNum in range(roundNumber):
        i = 1
        while not roundEndMarkArr[roundNum] - i in roundStartMarkArr:
            i += 1  # 找前面的开始点 以免中途打了别的标漏了起始点
        if roundEndMarkArr[roundNum] - i in roundStartMarkArr:  # 判断前面是不是round开始
            roundStartTime = eventTimeArr[roundEndMarkArr[roundNum] - 1]
            print("Round " + str(roundNum + 1) + ": roundStartTime -> " + str(roundStartTime))
            freq = raw.info.get('sfreq')  # 当前脑电数据的采样率
            """
            范式设计：
            实验起始打标（80） -> 3s
            -> 下一trial实验开始等待15s -> positive 1 情绪诱发视频30s
            -> 下一trial实验开始等待15s -> positive 2 情绪诱发视频30s
            -> 下一trial实验开始等待15s -> neutral 1 情绪诱发视频30s
            -> 下一trial实验开始等待15s -> neutral 2 情绪诱发视频30s
            -> 下一trial实验开始等待15s -> negative 1 情绪诱发视频30s
            -> 下一trial实验开始等待15s -> negative 2 情绪诱发视频30s
            -> 实验结束休息15s -> 实验结束打标（88）
            共288s 1000Hz->288000 200Hz->57600
            """
            roundTime = roundStartTime + freq * 3
            roundTime = roundTime + freq * 15
            round_trial = [
                't1_positive_1', 't2_positive_2', 't3_neutral_1', 't4_neutral_2', 't5_negative_1', 't6_negative_2'
            ]
            for trial_name in round_trial:
                trial_cnt_data = crop_cnt_data(raw_copy=raw.copy(), start=roundTime, duration_second=30)
                round_trial_name = 'r' + str(roundNum + 1) + '_' + trial_name
                # 保存每段trial的脑电
                save_data4pt(trial_cnt_data, DataFolder, subject_name, round_trial_name)
                roundTime = roundTime + freq * 30
                # 休息15秒
                roundTime = roundTime + freq * 15
        print(" ")


if __name__ == '__main__':
    data_folder = '../../Doc'
    get_CNT_path(data_folder)
