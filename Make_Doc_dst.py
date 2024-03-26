import os
import pickle
import mne
import torch
import numpy as np
from scipy.signal import resample


def make_1subject_dst(EEG_folder, ss_dst_folder):
    def get_label(file_name):
        global label
        state = file_name.split('_')[2]
        if state == 'negative':
            label = 0
        elif state == 'neutral':
            label = 1
        elif state == 'positive':
            label = 2
        return label

    # 脑电信号重采样，对齐SEED 1000Hz -> 200Hz
    def resample_data(data, origin_sfreq, target_sfreq):
        data_resampled = np.empty((data.shape[0], data.shape[1] // origin_sfreq * target_sfreq))
        for i in range(data.shape[0]):
            data_resampled[i, :] = resample(data[i, :], data.shape[1] // origin_sfreq * target_sfreq)
        return data_resampled

    nchn = 28
    nsec = 30
    origin_freq = 1000
    target_freq = 200
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]
    trial_files = [file for file in os.listdir(EEG_folder) if file.endswith('.pt')]
    os.makedirs(ss_dst_folder, exist_ok=True)

    subject_de = []
    subject_label = []
    # 对于每个trial
    for i, file_name in enumerate(trial_files):
        # 读取并重采样数据
        trial_data = torch.load(os.path.join(EEG_folder, file_name)).to(float).numpy()
        trial_data = resample_data(trial_data, origin_freq, target_freq)

        # 提取DE特征
        trial_de = np.zeros((nchn, nsec, len(freq_bands)))
        for j in range(len(freq_bands)):
            low_freq = freq_bands[j][0]
            high_freq = freq_bands[j][1]
            data_filt = mne.filter.filter_data(trial_data, target_freq, l_freq=low_freq, h_freq=high_freq)
            data_filt = data_filt.reshape(nchn, -1, target_freq)
            de_1band = 0.5 * np.log(2 * np.pi * np.exp(1) * (np.var(data_filt, 2)))
            trial_de[:, :, j] = de_1band
        trial_de = np.array([trial_de[:, idx:idx + 1, :] for idx in range(trial_de.shape[1])])
        subject_de.append(trial_de)
        subject_label = subject_label + [get_label(file_name)] * trial_de.shape[0]

    subject_de = np.vstack(subject_de)
    print("subject_de shape: ", subject_de.shape)
    subject_label = np.array(subject_label)
    print("subject_label shape: ", subject_label.shape)

    # 保存为pkl文件
    # with open(os.path.join(save_folder, subject, 'De_feature_data.pkl'), 'wb') as f:
    #     pickle.dump(subject_de, f)
    # with open(os.path.join(save_folder, subject, 'De_feature_label.pkl'), 'wb') as f:
    #     pickle.dump(subject_label, f)

    # 保存为npy文件
    np.save(os.path.join(ss_dst_folder, 'De_feature_data.npy'), subject_de)
    np.save(os.path.join(ss_dst_folder, 'De_feature_label.npy'), subject_label)


# continue
def make_LOSO_dst(DataFolder, LOSO_dst_name):
    subjects_folders = os.listdir(os.path.join(DataFolder, 'raw'))
    for subject_folder in subjects_folders:
        LOSO_dst_path = os.path.join(os.path.join(DataFolder, 'raw'), subject_folder, LOSO_dst_name)
        os.makedirs(LOSO_dst_path, exist_ok=True)
        dataset = {'train': [], 'test': []}
        label = {'train': [], 'test': []}
        for subject in subjects_folders:
            if subject == subject_folder:
                dataset['test'] = np.load(
                    os.path.join(DataFolder, 'raw', subject, 'SingleSubject_dst', 'De_feature_data.npy'))
                label['test'] = np.load(
                    os.path.join(DataFolder, 'raw', subject, 'SingleSubject_dst', 'De_feature_label.npy'))
            else:
                dataset['train'].append(np.load(
                    os.path.join(DataFolder, 'raw', subject, 'SingleSubject_dst', 'De_feature_data.npy')))
                label['train'].append(np.load(
                    os.path.join(DataFolder, 'raw', subject, 'SingleSubject_dst', 'De_feature_label.npy')))
        dataset['train'] = np.vstack(dataset['train'])
        label['train'] = np.hstack(label['train'])

        print("train data shape: ", dataset['train'].shape)
        print("train label shape: ", label['train'].shape)
        print("test data shape: ", dataset['test'].shape)
        print("test label shape: ", label['test'].shape)
        with open(os.path.join(LOSO_dst_path, 'train_data.pkl'), 'wb') as f:
            pickle.dump(dataset['train'], f)
        with open(os.path.join(LOSO_dst_path, 'train_label.pkl'), 'wb') as f:
            pickle.dump(label['train'], f)
        with open(os.path.join(LOSO_dst_path, 'test_data.pkl'), 'wb') as f:
            pickle.dump(dataset['test'], f)
        with open(os.path.join(LOSO_dst_path, 'test_label.pkl'), 'wb') as f:
            pickle.dump(label['test'], f)


if __name__ == '__main__':
    Doc_folder = '../Doc/EEG'
    save_folder = 'Doc_dst'
    make_1subject_dst(Doc_folder, save_folder)
