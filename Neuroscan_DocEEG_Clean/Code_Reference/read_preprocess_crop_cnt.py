# --------------------------------------------------------
# Read preprocess cnt files and crop trials
# Use read_preprocess_crop_cnt(subjects) to read and crop
# Written by Aisaka Aoi
# --------------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import scipy
import torch

# 只画图并保存，不显示图片 画图记得用plt.pause(0)
matplotlib.use('Agg')
# 在图片中显示中文、负号等
plt.rcParams['axes.unicode_minus'] = False  # 减号
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 中文字体
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['savefig.dpi'] = 300  # 图片像素


# 获取curry8预处理过的所有数据集cnt文件路径
def read_cnt_files(eeg_path):
    files_path = []
    for root, dirs, files in os.walk(eeg_path):
        if not root.endswith("Acquisition"):
            continue
        for file in files:
            if file.endswith("proc_convert.cdt.cnt"):
                files_path.append(os.path.join(root, file))
    return files_path


def detect_bad_channels_much_zero(raw_data, threshold=0.5):
    data = raw_data.get_data()
    zero_count = np.sum(data == 0, axis=1) / data.shape[1]
    bads = np.where(zero_count > threshold)[0]
    print('detect_bad_channels_much_zero: ', bads)
    return bads


def detect_bad_channels_avg_diff(raw_data, threshold_ratio=2e-3):
    data = raw_data.get_data()
    avg_diff = np.mean(np.abs(np.diff(data, axis=1)), axis=1)
    threshold = np.mean(avg_diff) * threshold_ratio
    bads = np.where(avg_diff < threshold)[0]
    print('detect_bad_channels_avg_diff: ', bads)
    return bads


def detect_bad_channels_variance(raw_data, threshold_factor=4):
    data = raw_data.get_data()
    variances = np.var(data, axis=1)
    mean_variance = np.mean(variances)
    std_variance = np.std(variances)
    bads = np.where((variances > mean_variance + threshold_factor * std_variance) | (
            variances < mean_variance - threshold_factor * std_variance))[0]
    print('detect_bad_channels_variance: ', bads)
    return bads


def detect_bad_channels_power_spectrum(raw_data, threshold_factor=2):
    data = raw_data.get_data()
    freqs, psds = scipy.signal.welch(data, fs=1000)
    # Compute mean and standard deviation of the power
    psd_mean = np.mean(psds, axis=1)
    psd_std = np.std(psds, axis=1)
    # Identify channels with abnormal power
    upper_threshold = psd_mean + threshold_factor * psd_std
    lower_threshold = psd_mean - threshold_factor * psd_std
    # Find channels where the mean power is outside the thresholds
    bad_channels_idx = np.where((psd_mean > upper_threshold) | (psd_mean < lower_threshold))[0]
    bads = [data.ch_names[i] for i in bad_channels_idx]
    print('detect_bad_channels_power_spectrum: ', bads)
    return bads


# 截取cnt脑电数据
def crop_cnt_data(raw_copy, start, duration_second):
    raw_copy.load_data()
    start_second = start / raw_copy.info['sfreq']
    end_second = start_second + duration_second - 1.0 / raw_copy.info['sfreq']
    # 获取指定tmin秒到tmax秒位置的eeg片段
    return raw_copy.crop(tmin=start_second, tmax=end_second)


def save_cnt_data(cnt_data, name, trial):
    data, _ = cnt_data[:, :]
    tensor_file = torch.Tensor(data)
    path_save = os.path.join("D:\\czn\\Datasets\\DoC_datas\\eeg", name)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    save_file = trial + ".pt"
    torch.save(tensor_file, path_save + '\\' + save_file)
    # print(torch.load(path_save + '\\' + save_file).shape)    # torch.Size([28, 30000])


def save_cnt_img(cnt_data, name, trial=None):
    if trial is None:
        path_save = os.path.join("D:\\czn\\Datasets\\DoC_datas\\img\\round", name)
    else:
        path_save = os.path.join("D:\\czn\\Datasets\\DoC_datas\\img\\trial", name, trial)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    # 绘制跨通道的功率谱密度图
    cnt_data.compute_psd(fmin=0, fmax=4).plot()
    plt.savefig(path_save + '\\plot_psd_0_4.jpg')
    cnt_data.compute_psd(fmin=4, fmax=8).plot()
    plt.savefig(path_save + '\\plot_psd_4_8.jpg')
    cnt_data.compute_psd(fmin=8, fmax=13).plot()
    plt.savefig(path_save + '\\plot_psd_8_13.jpg')
    cnt_data.compute_psd(fmin=13, fmax=30).plot()
    plt.savefig(path_save + '\\plot_psd_13_30.jpg')
    cnt_data.compute_psd(fmin=30, fmax=50).plot()
    plt.savefig(path_save + '\\plot_psd_30_50.jpg')
    cnt_data.compute_psd(fmin=0, fmax=50).plot()
    plt.savefig(path_save + '\\plot_psd_0_50.jpg')
    # 绘制EEG时域波形
    data, times = cnt_data[:]
    duration = len(times) / cnt_data.info.get('sfreq')
    cnt_data.plot(start=0, duration=duration)
    plt.savefig(path_save + '\\all_cnt.jpg')
    # 频率分析
    # cnt_data.plot_psd_topomap(normalize=True)
    # plt.savefig(path_save + '\\plot_psd_topomap_normalize_True.jpg')
    # cnt_data.plot_psd_topomap(normalize=False)
    # plt.savefig(path_save + '\\plot_psd_topomap_normalize_False.jpg')
    # 关闭绘图 释放内存
    plt.close()


def read_preprocess_crop_cnt(subjects):
    # cnt_files_path = read_cnt_files("D:\\czn\\Datasets\\DoC_datas\\raw")
    # print(len(cnt_files_path), cnt_files_path)
    cnt_files_path = []
    for subject in subjects:
        cnt_files_path.append(
            "D:\\czn\\Datasets\\DoC_datas\\raw\\" + subject + "\\Acquisition\\Acquisition 01_proc_convert.cdt.cnt")

    for index, cnt_file_path in enumerate(cnt_files_path):
        print('=' * 80)
        print('=' * 80)
        print("Subject " + str(index) + ": " + cnt_file_path.split('\\')[-3])
        # 读取数据
        raw = mne.io.read_raw_cnt(cnt_file_path, preload=True)
        # ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4',
        #  'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'Pz', 'P4', 'T6', 'A2', 'O1', 'Oz', 'O2']
        """
        # 通常raw的数据访问方式如下：data, times = raw[picks, time_slice]
        # 若需访问raw中所有数据，代码如下：data, times = raw[:]
        # 或者[:, :]
        print(type(raw))  # 查看raw类型
        print(raw.pick(['eeg']))
        print(raw.info)
        print(raw.info.get('ch_names'))
        print(raw.info.get('bads'))
        print(raw.info.get('nchan'))
        print(raw.info.get('sfreq'))  # 采样频率
        print(raw.get_data().shape)  # 获取数据的行列
        print(raw.info['ch_names'])  # 打印通道名
        """

        # 删除无用通道
        raw.drop_channels(['HEOL', 'VEOU', 'HEOR-L', 'VEOL-U', 'Trigger', 'A1', 'A2', 'TP7', 'TP8'])
        print("After drop_channels remains: " + str(raw.info['nchan']) + " channels")  # 通道数

        # 如果需要电极重参考
        """
        # 默认小放大器已经是用Fz参考
        raw.set_eeg_reference(ref_channels=['Fz'])
        # 比如采样A1 A2两个乳突电极进行重参考
        raw.set_eeg_reference(ref_channels=["A1", "A2"])  # 基于A1 A2完成重参考
        # 或者平均参考
        raw.set_eeg_reference(ref_channels='average')
        raw.plot(start=20, duration=1, n_channels=64, block=True, title='重参考完成，无误请关闭窗口')
        """

        # 如果有坏导
        bad_channels = []
        # 旧的手动检测办法
        bad_channels.extend(detect_bad_channels_much_zero(raw))
        bad_channels.extend(detect_bad_channels_avg_diff(raw))
        bad_channels.extend(detect_bad_channels_variance(raw))
        bad_channels.extend(detect_bad_channels_power_spectrum(raw))
        bad_channels = list(set(bad_channels))
        result = [raw.info['ch_names'][i] for i in bad_channels]
        print('所有坏道bad_channels: ', result)
        raw.info['bads'].extend(result)  # 确定坏导
        if raw.info['bads']:
            raw.interpolate_bads()

        # 环境滤波: 陷波滤波
        raw = raw.notch_filter(freqs=50)
        raw = raw.notch_filter(freqs=60)
        # 高低通滤波
        raw = raw.filter(l_freq=0.1, h_freq=50)  # ⾼通滤波为了消除电压漂移，低通滤波为了消除⾼频噪⾳

        # ICA独立成分分析
        ica = mne.preprocessing.ICA(n_components=20)
        # 先fit一下查看有哪些伪影
        ica.fit(raw, reject_by_annotation=False)
        # ica.plot_sources(raw_ica, show_scrollbars=False)
        # 然后apply一下就可以应用到原始数据了
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=raw.info['ch_names'][0], reject_by_annotation=False)
        mus_indices, mus_scores = ica.find_bads_muscle(raw)
        artifacts_indices = eog_indices + mus_indices
        ica.exclude = artifacts_indices
        raw = ica.apply(raw, exclude=ica.exclude)
        # ica.plot_sources(raw_ica, show_scrollbars=False)
        # ica.plot_components()
        # ICA分析,这个是单独拿成分出来 画出这15个ICA的脑电图
        # ica.plot_properties(raw_ica, picks=ica.exclude)

        # EEG重采样
        # raw.resample(200)

        # 预处理后的整段脑电绘图
        subject_name = cnt_file_path.split('\\')[-3]  # 从文件路径找到人名
        # save_cnt_img(raw, subject_name)

        # 检测标签
        # 找到并裁剪出我们需要的片段，比如说标记了的事件相关信号
        events = mne.events_from_annotations(raw)  # 将raw里面的annotation读取出来
        # print(events)
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
                    save_cnt_data(trial_cnt_data, subject_name, round_trial_name)
                    # 保存每段trial的图片
                    # save_cnt_img(trial_cnt_data, subject_name, round_trial_name)
                    # print(round_trial_name + ': ' + str(roundTime) + ' -> ' + str(roundTime + freq * 30), end=" ")
                    roundTime = roundTime + freq * 30
                    # 休息15秒
                    roundTime = roundTime + freq * 15
            print(" ")
