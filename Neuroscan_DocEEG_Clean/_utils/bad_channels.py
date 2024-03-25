import mne
import numpy as np
import scipy
from autoreject import AutoReject
from mne.io import RawArray


class BadChanInterp:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    # 坏道检测及插值修复
    def detect_and_interp_byAoi(self):
        bad_channels = []
        bad_channels.extend(self.detect_bad_channels_much_zero())
        bad_channels.extend(self.detect_bad_channels_avg_diff())
        bad_channels.extend(self.detect_bad_channels_variance())
        bad_channels.extend(self.detect_bad_channels_power_spectrum())
        bad_channels = list(set(bad_channels))
        result = [self.raw_data.info['ch_names'][i] for i in bad_channels]
        print('All detected bad channels: ', result)
        # 确定坏导并插值
        self.raw_data.info['bads'].extend(result)
        if self.raw_data.info['bads']:
            self.raw_data.interpolate_bads()
        return self.raw_data

    def detect_and_interp_byJerry(self):
        # 定义时间段的长度和重叠（根据需求进行调整）
        duration = 10.0  # 时间段的长度（单位：秒）
        overlap = 0.0  # 重叠部分的比例

        # 获取原始数据的采样频率
        sfreq = self.raw_data.info['sfreq']
        # 计算时间段的采样点数和重叠的采样点数
        n_samples = int(duration * sfreq)
        n_overlap = int(n_samples * overlap)

        # 创建虚拟事件，基于时间段的分割
        events = mne.make_fixed_length_events(self.raw_data, duration=duration, overlap=n_overlap)
        epochs = mne.Epochs(self.raw_data, events, tmin=0, tmax=duration, baseline=None, preload=True)

        # 坏道插值重建
        ar = AutoReject()
        epochs_clean = ar.fit_transform(epochs)  # doctest: +SKIP

        epochs_data = epochs_clean.get_data().transpose((1, 0, 2))
        new_raw_data = []
        for lead in range(epochs_data.shape[0]):
            new_lead_data = []
            for epoch in range(epochs_data.shape[1]):
                new_lead_data.extend(epochs_data[lead, epoch, :])
            new_raw_data.append(new_lead_data)
        new_raw_data = np.array(new_raw_data)
        raw = RawArray(new_raw_data, self.raw_data.info)
        picks = mne.pick_types(
            self.raw_data.info, meg=False, eeg=True, stim=False,
            include=self.raw_data.info['ch_names']
        )
        raw.save("raw.fif", picks=picks, overwrite=True)
        raw = mne.io.read_raw_fif("raw.fif", preload=True, verbose='ERROR')
        return raw

    # 零值的比例超过了给定的阈值即为坏道
    def detect_bad_channels_much_zero(self, threshold=0.5):
        data = self.raw_data.get_data()
        zero_count = np.sum(data == 0, axis=1) / data.shape[1]
        bads = np.where(zero_count > threshold)[0]
        print('Bad channels detected using much zero method: ', bads)
        return bads

    # 通道平均差分低于整体平均差分一定比例的即为坏道
    def detect_bad_channels_avg_diff(self, threshold_ratio=2e-3):
        data = self.raw_data.get_data()
        avg_diff = np.mean(np.abs(np.diff(data, axis=1)), axis=1)
        threshold = np.mean(avg_diff) * threshold_ratio
        bads = np.where(avg_diff < threshold)[0]
        print('Bad channels detected using avg diff method: ', bads)
        return bads

    # 处于正常方差范围外的通道即为坏道
    def detect_bad_channels_variance(self, threshold_factor=4):
        data = self.raw_data.get_data()
        variances = np.var(data, axis=1)
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        bads = np.where((variances > mean_variance + threshold_factor * std_variance) | (
                variances < mean_variance - threshold_factor * std_variance))[0]
        print('Bad channels detected using variance method: ', bads)
        return bads

    # psd均值超出范围的通道即为坏道
    def detect_bad_channels_power_spectrum(self, threshold_factor=2):
        data = self.raw_data.get_data()
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
        print('Bad channels detected using power spectrum method: ', bads)
        return bads
