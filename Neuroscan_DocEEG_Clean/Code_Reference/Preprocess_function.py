import mne
import numpy as np
import os
import shutil
from scipy.signal.windows import hann
from scipy import stats
from mne.preprocessing import ICA
import pickle as pkl
import re
import math
from collections import Counter

# center_channel_info:
Neuroscan31 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
               'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M1', 'TP7', 'CP3',
               'CP4', 'TP8', 'M2', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'O1', 'Oz',
               'O2']

Neuroscan29 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz',
               'FC4', 'FT8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP7', 'CP3', 'CP4',
               'TP8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'O1', 'Oz', 'O2']


# 博睿康数据的预处理
class Neuracle_Preprocessing():

    def __init__(self, raw):
        # Modify the montage
        self.device = 'Neuracle31'
        self.nchns = raw.info['nchan']
        self.freq = raw.info['sfreq']
        # 如果有需要调整名称的通道
        old_chn_names = raw.info['ch_names']
        new_chn_names = raw.info['ch_names'].copy()
        chan_names_dict = {old_chn_names[i]: new_chn_names[i] for i in range(len(old_chn_names))}
        raw.rename_channels(chan_names_dict)
        montage = mne.channels.make_standard_montage('standard_1020')
        print('DIGI MONTAGE:', montage)
        raw.set_montage(montage)
        # Match the corresponding montage to their index
        self.montage_index = dict(zip(np.arange(self.nchns), new_chn_names))

        # split out the data matrix
        self.raw = raw
        # Ptr operation
        self.data = self.raw.get_data()

    def plot_eeg(self, second):
        self.raw.plot(duration=second, n_channels=32, clipping=None)

    def plot_sensors(self):
        self.raw.plot_sensors(ch_type='eeg', show_names=True)

    def band_pass_filter(self, l_freq, h_freq):
        # The default filter is a FIR filter,
        # therefore,  the input [l_freq,h_freq] is
        # [lower pass-band edge, higher pass-band edge]
        self.raw.filter(l_freq, h_freq)

    def down_sample(self, n_freq):
        # 我不知道为什么它自己没有降采样，得我来降
        self.raw.resample(n_freq)
        # eeg_data = self.raw.get_data()
        # print('from shape ', str(self.raw.info['sfreq']),' to ', str(n_freq))
        # size = int(self.raw.info['sfreq']/n_freq)
        # down_sample_data = eeg_data[:, ::size]
        # self.raw._data = down_sample_data
        # # self.raw.info['sfreq'] = n_freq
        self.freq = n_freq

    # Should cut the data into pieces then do the interpolation?
    def bad_channels_interpolate(self, thresh1=None, thresh2=None, proportion=0.3):
        data = self.raw.get_data()
        # We found that the data shape of epochs is 3 dims
        # print(data.shape)
        if len(data.shape) > 2:
            data = np.squeeze(data)
        Bad_chns = []
        value = 0
        # Delete the much larger point
        if thresh1 != None:
            md = np.median(np.abs(data))
            value = np.where(np.abs(data) > (thresh1 * md), 0, 1)[0]
        if thresh2 != None:
            value = np.where((np.abs(data)) > thresh2, 0, 1)[0]
        # Use the standard to pick out the bad channels
        Bad_chns = np.argwhere((np.mean((1 - value), axis=0) > proportion))
        if Bad_chns.size > 0:
            self.raw.info['bads'].extend([self.montage_index[str(bad)] for bad in Bad_chns])
            print('Bad channels: ', self.raw.info['bads'])
            self.raw = self.raw.interpolate_bads()
        else:
            print('No bad channel currently')

    # You can manually exclude the ICA elements
    # Our auto preprocessing method is more effective for eye blink removal
    def eeg_ica(self, check_ica=None):
        ica = ICA(n_components=len(self.raw.ch_names), max_iter='auto', method='fastica')
        raw_ = self.raw.copy()
        ica.fit(self.raw)
        # Plot different elements of the signals
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name='Fp1')
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name='Fp2')
        eog_indices = list(set(eog_indices1 + eog_indices2))
        ica.exclude = eog_indices
        # ica.plot_sources(raw_)
        # Plot the components
        # ica.plot_components()
        if check_ica == True:
            print('Already use:', eog_indices)
            ica.plot_sources(raw_)
            ica.plot_components()
            eog_indices = input("Exclude ?")
            eog_indices = eog_indices.split(" ")
            eog_indices = list(map(int, eog_indices))
            ica.exclude = eog_indices
        ## Plot excluded one of the elements
        # ica.plot_overlay(raw_, exclude=[1])
        # ica.plot_properties(raw_, picks=[1, 16])

        # Exclude the elements you don't want
        ica.apply(self.raw)
        # self.raw.plot(duration = 5,n_channels = self.nchns,clipping = None)

    def delete_ref(self, ref_chn):
        if ref_chn == 'no':
            # 删除参考通道
            for chn in ['A1', 'A2', 'HEOL', 'HEOR', 'Trigger']:
                if chn in self.raw.info['ch_names']:
                    self.raw.drop_channels(chn)
            print("删除后的通道名称：", self.raw.ch_names)
            self.device = 'Neuracle29'
        else:
            pass

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')

    def _get_average_psd(self, energy_graph, freq_bands, sample_freq, stft_n=256):
        start_index = int(np.floor(freq_bands[0] / sample_freq * stft_n))
        end_index = int(np.floor(freq_bands[1] / sample_freq * stft_n))
        ave_psd = np.mean(energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
        return ave_psd

    def extract_psd_feature(self, window_size, freq_bands, stft_n=256):
        sample_freq = self.raw.info['sfreq']
        # Ptr operation
        self.data = self.raw.get_data()
        if len(self.data.shape) > 2:
            self.data = np.squeeze(self.data)
        n_channels, n_samples = self.data.shape
        point_per_window = int(sample_freq * window_size)
        window_num = int(n_samples // point_per_window)
        psd_feature = np.zeros((window_num, len(freq_bands), n_channels))

        for window_index in range(window_num):
            start_index, end_index = point_per_window * window_index, point_per_window * (window_index + 1)
            window_data = self.data[:, start_index:end_index]
            hdata = window_data * hann(point_per_window)
            fft_data = np.fft.fft(hdata, n=stft_n)
            energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])

            for band_index, band in enumerate(freq_bands):
                band_ave_psd = self._get_average_psd(energy_graph, band, sample_freq, stft_n)
                psd_feature[window_index, band_index, :] = band_ave_psd
        return psd_feature

    # 调整通道顺序至与第一批数据相同
    def channel_modify(self):
        current_order = self.raw.ch_names
        new_order = []
        print('device name:', self.device)
        if self.device == 'Neuracle29':
            # 更改旧的通道里原来通道没有的名称
            rename_dict = {'T3': 'T7', 'T4': 'T8', 'T5': 'PO7', 'T6': 'PO8'}
            self.raw.rename_channels(rename_dict)
            new_order = Neuroscan29
            print(self.raw.ch_names)
        # 确保新的顺序列表包含了所有的通道
        # assert set(new_order) == set(current_order), "新的通道顺序列表必须与当前通道列表匹配"
        self.raw.reorder_channels(new_order)


# Neuroscan数据的预处理
class Neuroscan_Preprocessing():

    def __init__(self, raw):
        self.device = 'Neuroscan33'
        '''
        # Neuroscan
        ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T7', 'C3', 'CZ', 'C4', 'T8', 
        'M1', 'TP7', 'CP3', 'CP4', 'TP8', 'M2', 'P3', 'PZ', 'P4', 'PO7', 'PO8', 'O1', 'OZ', 'O2', 'HEO', 'Trigger']
        '''
        self.nchns = raw.info['nchan']  # 33
        self.freq = raw.info['sfreq']
        old_chn_names = raw.info['ch_names']
        new_chn_names = raw.info['ch_names'].copy()
        # 重命名！
        if self.device == 'Neuroscan33':
            # 更改旧的通道里原来通道没有的名称
            rename_dict = {'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz'}
            raw.rename_channels(rename_dict)
        montage = mne.channels.make_standard_montage('standard_1020')
        print('DIGI MONTAGE:', montage)
        raw.set_montage(montage)
        # Match the corresponding montage to their index
        self.montage_index = dict(zip(np.arange(self.nchns), new_chn_names))
        # split out the data matrix
        self.raw = raw
        # Ptr operation
        self.data = self.raw.get_data()

    def plot_eeg(self, second):
        self.raw.plot(duration=second, n_channels=32, clipping=None)

    def plot_sensors(self):
        self.raw.plot_sensors(ch_type='eeg', show_names=True)

    def band_pass_filter(self, l_freq, h_freq):
        # The default filter is a FIR filter,
        # therefore,  the input [l_freq,h_freq] is
        # [lower pass-band edge, higher pass-band edge]
        self.raw.filter(l_freq, h_freq)

    def down_sample(self, n_freq):
        # 我不知道为什么它自己没有降采样，得我来降
        self.raw.resample(n_freq)
        # eeg_data = self.raw.get_data()
        # print('from shape ', str(self.raw.info['sfreq']),' to ', str(n_freq))
        # size = int(self.raw.info['sfreq']/n_freq)
        # down_sample_data = eeg_data[:, ::size]
        # self.raw._data = down_sample_data
        # # self.raw.info['sfreq'] = n_freq
        self.freq = n_freq

    # Should cut the data into pieces than do the interpolation
    def bad_channels_interpolate(self, thresh1=None, thresh2=None, proportion=0.3):
        data = self.raw.get_data()
        # We found that the data shape of epochs is 3 dims
        # print(data.shape)
        if len(data.shape) > 2:
            data = np.squeeze(data)
        Bad_chns = []
        value = 0
        # Delete the much larger point
        if thresh1 != None:
            md = np.median(np.abs(data))
            value = np.where(np.abs(data) > (thresh1 * md), 0, 1)[0]
        if thresh2 != None:
            value = np.where((np.abs(data)) > thresh2, 0, 1)[0]
        # Use the standard to pick out the bad channels
        Bad_chns = np.argwhere((np.mean((1 - value), axis=0) > proportion))
        if Bad_chns.size > 0:
            self.raw.info['bads'].extend([self.montage_index[str(bad)] for bad in Bad_chns])
            print('Bad channels: ', self.raw.info['bads'])
            self.raw = self.raw.interpolate_bads()
        else:
            print('No bad channel currently')

    # You can manually exclude the ICA elements
    # Our auto preprocessing method is more effective for eye blink removal
    def eeg_ica(self, check_ica=None):
        ica = ICA(n_components=len(self.raw.ch_names), max_iter='auto', method='fastica')
        raw_ = self.raw.copy()
        ica.fit(self.raw)
        # Plot different elements of the signals
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name='Fp1')
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name='Fp2')
        eog_indices = list(set(eog_indices1 + eog_indices2))
        ica.exclude = eog_indices
        # ica.plot_sources(raw_)
        # Plot the components
        # ica.plot_components()
        if check_ica == True:
            print('Already use:', eog_indices)
            ica.plot_sources(raw_)
            ica.plot_components()
            eog_indices = input("Exclude ?")
            eog_indices = eog_indices.split(" ")
            eog_indices = list(map(int, eog_indices))
            ica.exclude = eog_indices
        ## Plot excluded one of the elements
        # ica.plot_overlay(raw_, exclude=[1])
        # ica.plot_properties(raw_, picks=[1, 16])

        # Exclude the elements you don't want
        ica.apply(self.raw)
        # self.raw.plot(duration = 5,n_channels = self.nchns,clipping = None)

    def delete_ref(self, ref_chn):
        if ref_chn == 'no':
            # 删除参考通道
            for chn in ['A1', 'A2', 'HEOL', 'HEOR', 'HEO', 'M1', 'M2', 'Trigger']:  # Trigger信号已经提前取走
                if chn in self.raw.info['ch_names']:
                    self.raw.drop_channels(chn)
            print("删除后的通道名称：", self.raw.ch_names)
            self.device = 'Neuroscan29'
        else:
            pass

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')


# 数据降采样 -> 一般是降到250
def trigger_downsample(trigger, freq, target_freq):
    # 1000 -> 250
    size = freq / target_freq
    trigger = trigger.reshape(-1)
    down_sample_trigger = np.zeros((int(trigger.shape[0] / size)))
    trigger_pos = np.where(trigger != 0)[0]
    # print(trigger_pos)
    for trigger_idx in trigger_pos:
        value = trigger[trigger_idx]
        down_sample_trigger_idx = int(trigger_idx / 4)
        down_sample_trigger[down_sample_trigger_idx] = value
    down_sample_trigger = down_sample_trigger.reshape(1, -1)
    print('new trigger shape:', down_sample_trigger.shape)
    return down_sample_trigger
