from Make_Doc_dst import make_1subject_dst, make_LOSO_dst
from Neuroscan_DocEEG_Clean.main import preprocess_main

if __name__ == '__main__':
    # preprocess
    OriginData_Folder = './Doc'
    ICA_Folder = './preprocess_img/ICA'
    PSD_Folder = './preprocess_img/PSD'
    preprocess_main(OriginData_Folder, ICA_Folder, PSD_Folder)

    # 为每一个被试单独制作数据集用以预测
    EEG_Folder = './Doc/EEG'
    ss_Folder = './Doc_SingleSubject'
    make_1subject_dst(EEG_Folder, ss_Folder)

    # 制作留一被试数据集
    LOSO_Folder = './Doc_LOSO'
    # make_LOSO_dst(ss_Folder, LOSO_Folder)
