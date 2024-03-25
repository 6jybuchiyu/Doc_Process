# Authors: Robert Wong <robert_is_here@163.com>
# Date : 2024.2.27
# This code is used for preprocess the eeg data from cable Neuroscan EEG device with 33 channels

import mne
import numpy as np
import os
import shutil
import argparse
from Preprocess_function import *
from read_data import *
import pickle as pkl
import re
import math
from collections import Counter

# 2024-02 博睿康有线设备31预处理流程
# 脑电 -> 滤波 -> 坏通道插值 -> 自动ICA -> 去掉参考电极 -> 全局平均
parser = argparse.ArgumentParser(description='Preprocess the eeg data')

parser.add_argument('--device', default='neuroscan', type=str, help='Neuracle or Neuroscan or BP device')

parser.add_argument('--bp_low', default=0.05, type=float, help='the low freq of the band pass filter')
parser.add_argument('--bp_high', default=47, type=float, help='the high freq of the band pass filter')

# 记住，这里的插值只是为了解决坏掉的通道的问题，无法解决运动噪声
parser.add_argument('--inter', default=1, type=int, help='whether to inter the data.If inter = 0 ,dont inter.')
parser.add_argument('--bad_ratio', default=0.3, type=float,
                    help='the ratio to define the bad channel in the data. 0 means without channel inter')
parser.add_argument('--A1A2', default='no', type=str,
                    help='whether to delete the A1A2/M1M2 channel. No means delete it.')  # M1M2 is different from A1A2

parser.add_argument('--ICA', default='yes', type=str, help='whether to use auto ICA.')

parser.add_argument('--average', default='common', type=str,
                    help='common average or A1A2 average')  # M1M2 is deleted before average

args = parser.parse_args()
device = args.device
low_freq = args.bp_low
high_freq = args.bp_high
inter = args.inter
bad_ratio = args.bad_ratio
A1A2 = args.A1A2
ica = args.ICA
target_freq = 250

neuroscan_dir = '../../RAW_DATA/Neuroscan31_HC1_74/'
neuroscan__files = os.listdir(neuroscan_dir)
neuroscan__files.sort()
# print(files)
save_dir = '../../Fusion_preprocessed_data/EEG/eeg_inter{}_bp{}_{}_ICA_{}_ref_{}'.format(bad_ratio, low_freq, high_freq,
                                                                                         ica, A1A2)

from datetime import datetime

# 获取当前时间
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open('neuroscan_eeg_record.txt', 'a') as f:
    f.write("当前时间: %s\n" % current_time)
    f.write("当前参数： %s\n" % save_dir)

'''
    "Curry 8": {
        "info": ".cdt.dpa",
        "data": ".cdt",
        "labels": ".cdt.dpa",
        "events_cef": ".cdt.cef",
        "events_ceo": ".cdt.ceo",
        "hpi": ".cdt.hpi",
    }
'''

for sub_eeg in neuroscan__files:
    print('processing sub:', sub_eeg)
    sub_file = os.path.join(neuroscan_dir, sub_eeg, 'EEG')
    # cdt的范式有些主试喜欢断开实验
    sub_data_files = [file for file in os.listdir(sub_file) if file.endswith('cdt')]

    for data_file in sub_data_files:
        sub_data_dpa = os.path.join(sub_file, data_file + '.dpa')
        # 修改数据至符合MNE格式
        sub_data_dpo = os.path.join(sub_file, data_file + '.dpo')
        shutil.copy(sub_data_dpo, sub_data_dpa)
        print('copy:', sub_data_dpa)
        data_file_ = os.path.join(sub_file, data_file)
        print(data_file)
        # 保存路径
        save_sub = os.path.join(save_dir, sub_eeg, 'EEG')  # 需要创建
        # savepath = os.path.join(sub_file, data_file[:-4] + '.pkl')

        # 加载原始脑电数据
        rawdata = mne.io.read_raw_curry(data_file_, preload=True)
        data = rawdata.get_data()  # with Trigger channel
        print('original data shape:', data.shape)

        frequency = rawdata.info['sfreq']
        print('frequney of original data:', str(frequency))
        chname = rawdata.ch_names
        print(chname)

        ######## 处理trigger, 然后调整单位
        events, _ = mne.events_from_annotations(rawdata)
        print('THE EVENT RULE IN THE EXPERIMENT:', _)
        print('event shape:', events.shape, events[0, :])
        onset = np.array([int(x) for x in events[:, 0]])
        duration = np.array([int(x) for x in events[:, 1]])

        try:

            trigger = np.array([i for i in events[:, 2]])
            # print('Old trigger:',trigger)
            # 置换为原本的字典格式
            restored_array = [key for value in trigger for key, dict_value in _.items() if dict_value == value]
            new_trigger = np.array([int(x) for x in restored_array])
            # print('NEW trigger:',new_trigger)
            events[:, 2] = new_trigger

        except Exception as e:

            print('MAYBE STRING IN THE TIGGER CHANNEL')
            print('错误类型:', e.__class__.__name__, '错误明细:', e)
            raise ValueError("trigger转换过程中有问题，请检查报错")

        # 处理event
        eventCode = []
        eventPosition = []

        for eventsIndex in range(len(events)):
            eventPosition.append(events[eventsIndex][0])
            eventCode.append(events[eventsIndex][2])

        triggerChannel = np.zeros((1, data.shape[1]))
        triggerChannel[0][eventPosition] = eventCode

        # trigger通道已经done，卸下来评估一下单位
        rawdata.drop_channels('Trigger')
        rawdata, unit, data_mean = unit_check(rawdata)

        # 预处理
        # ——————————通道需要调整为正规的31导通道————————————

        processed_raw = Neuroscan_Preprocessing(rawdata)
        processed_raw.delete_ref(A1A2)  # 31->29 # 为了ICA更安全，先把trigger通道扔掉
        processed_raw.down_sample(target_freq)
        # processed_raw.band_pass_filter(0.5, 47)
        processed_raw.band_pass_filter(0.05, 47)
        processed_raw.bad_channels_interpolate(thresh1=3, proportion=bad_ratio)
        if ica == 'yes':
            processed_raw.eeg_ica()
        processed_raw.average_ref()

        eeg_data = processed_raw.raw.get_data()

        print('EEG DATA SHAPE:', eeg_data.shape)
        print('TRIGGER SHAPE:', triggerChannel.shape)

        triggerChannel = trigger_downsample(triggerChannel, frequency, target_freq)
        print('downsampled trigger shape:', triggerChannel.shape)

        eeg = np.vstack((eeg_data[:, :], triggerChannel))
        print('FINAL EEG SHAPE:', eeg.shape)

        if not os.path.exists(save_sub):
            # 文件夹不存在，创建文件夹
            os.makedirs(save_sub)
            print(f"Build '{save_sub}'")
        else:
            # 文件夹已存在
            print(f"'{save_sub}' already exists")

        savepath = os.path.join(save_sub, data_file[:-4] + '.pkl')

        fi = open(savepath, 'wb')
        pkl.dump(eeg, fi)
        fi.close()
        print('SAVED:', savepath)

        with open('neuroscan_eeg_record.txt', 'a') as f:
            # 是不是Neuroscan设备没有中途查看阻抗的情况？
            print(set(list(triggerChannel[0])))
            # '100255'这种奇怪的trigger可能就是了
            f.write("当前数据结果： {} 单位：{}_{}\n".format(sub_eeg, unit, data_mean))
