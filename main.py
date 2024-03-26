from Make_Doc_dst import make_1subject_dst, make_LOSO_dst
from Neuroscan_DocEEG_Clean.main import preprocess_main

if __name__ == '__main__':
    # 这里的ICA_folder_name和PSD_folder_name是每个被试文件夹下存访ICA和PSD图片的文件夹名称，而不是什么路径
    preprocess_main(DataFolder='./Doc', ICA_folder_name='ICA', PSD_folder_name='PSD')

    # # 制作留一被试数据集，同样地，这里LOSO_dst_name也是文件夹名称，而不是路径
    # make_LOSO_dst(DataFolder='./Doc', LOSO_dst_name='LOSO_dst')
