import os
from Neuroscan_DocEEG_Clean._utils.bad_channels import BadChanInterp
from Neuroscan_DocEEG_Clean._utils.ica import run_ica
from Neuroscan_DocEEG_Clean._utils.utils import get_CNT_path, read_data, plot_psd, split_data


def preprocess_main(DataFolder, ICA_folder, PSD_folder):
    # get all subjects' cnt files path
    CNT_Paths = get_CNT_path(DataFolder)

    # to save the ICA and PSD images
    os.makedirs(ICA_folder, exist_ok=True)
    os.makedirs(PSD_folder, exist_ok=True)

    for idx, path in enumerate(CNT_Paths):
        subject_name = path.split('\\')[-3]

        # 读取数据+初步预处理
        raw = read_data(path)

        # 坏道检测+插值重建
        ChanInterp = BadChanInterp(raw)
        raw = ChanInterp.detect_and_interp_byAoi()

        # ICA伪迹去除
        raw = run_ica(raw, ICA_folder)
        plot_psd(raw, PSD_folder)

        # 数据分段和保存
        split_data(raw, subject_name, DataFolder)


if __name__ == '__main__':
    DataFolder = '../Doc'
    ICA_folder = './img/ICA'
    PSD_folder = './img/PSD'
    preprocess_main(DataFolder, ICA_folder, PSD_folder)
