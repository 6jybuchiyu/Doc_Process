import os

from utils.bad_seg_clean import bad_seg_clean
from utils.ica_clean import ica_clean
from utils.load_data import load_data, get_data_files_path
from utils.save_data import save_data
from utils.show_img import show_img

if __name__ == "__main__":
    files_path = get_data_files_path("../Doc")
    os.makedirs("img/ICA/", exist_ok=True)
    os.makedirs("img/PSD/", exist_ok=True)
    bands = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 13], 'beta': [13, 30], 'gamma': [30, 100]}
    for file_path in files_path:
        origin_raw = load_data(file_path)
        raw = origin_raw.copy()
        show_img(raw, 100)

        # 坏道检测 插值
        # raw = bad_channels_clean(raw, raw.info, raw.info.ch_names)

        # 坏段拒绝
        # 设置EOG事件的开始时间和持续时长(500ms)等
        raw = bad_seg_clean(raw, thresh=70)
        # show_img(raw, 100)
        # show_img(raw, 100)

        # ICA 伪迹清除
        raw = ica_clean(raw)
        # show_img(raw, 100)

        save_data(raw, "ProcessedByJin", file_path)
