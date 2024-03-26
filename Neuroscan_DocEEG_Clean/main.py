import os
from Make_Doc_dst import make_1subject_dst
from Neuroscan_DocEEG_Clean._utils.bad_channels import BadChanInterp
from Neuroscan_DocEEG_Clean._utils.ica import run_ica
from Neuroscan_DocEEG_Clean._utils.utils import get_CNT_path, read_data, plot_psd, split_data, get_subject_path


def preprocess_main(DataFolder, ICA_folder_name, PSD_folder_name):
    # get all subjects' cnt files path
    CNT_Paths = get_CNT_path(DataFolder)

    for idx, path in enumerate(CNT_Paths):
        subject_name = path.split('\\')[-3]
        subject_path = get_subject_path(subject_name, path)

        # to save the ICA and PSD images
        os.makedirs(os.path.join(subject_path, 'preprocess_img', ICA_folder_name), exist_ok=True)
        os.makedirs(os.path.join(subject_path, 'preprocess_img', PSD_folder_name), exist_ok=True)

        # 读取数据+初步预处理
        raw = read_data(path)

        # 坏道检测+插值重建
        ChanInterp = BadChanInterp(raw)
        raw = ChanInterp.detect_and_interp_byAoi()

        # ICA伪迹去除
        raw = run_ica(raw, os.path.join(subject_path, 'preprocess_img', ICA_folder_name))
        plot_psd(raw, os.path.join(subject_path, 'preprocess_img', PSD_folder_name))

        # 数据分段和保存
        split_data(raw, subject_name, DataFolder)

        # 为每一个被试单独制作数据集用以预测
        EEG_Folder = os.path.join(subject_path, 'EEG')
        ss_Folder = os.path.join(subject_path, 'SingleSubject_dst')
        make_1subject_dst(EEG_Folder, ss_Folder)


if __name__ == '__main__':
    DataFolder = '../Doc'
    ICA_folder = './img/ICA'
    PSD_folder = './img/PSD'
    preprocess_main(DataFolder, ICA_folder, PSD_folder)
