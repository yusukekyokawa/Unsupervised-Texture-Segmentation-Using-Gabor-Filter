import cv2
import glob
import os
from tqdm import tqdm

def resize_image():
    # リサイズ
    ROOT = "../型紙データ20171005_ARC"
    ONE_THIRD_ROOT = "../RESIZE/ONE_THIRD/"
    QUARTER_ROOT = "../RESIZE/QUARTER/"

    folderlist = glob.glob(ROOT + "/arcKG*")
    for folder in tqdm(folderlist):
        foldername = os.path.basename(folder)
        one_third_folder = os.path.join(ONE_THIRD_ROOT, foldername)
        quarter_folder = os.path.join(QUARTER_ROOT, foldername)
        # 画像のフォルダ作成
        os.makedirs(one_third_folder, exist_ok=True)
        os.makedirs(quarter_folder, exist_ok=True)
        pathlist = sorted(glob.glob(folder + "/*.jpg"))
        for path in pathlist:    
            filename = os.path.basename(path)
            ori_img = cv2.imread(path)
            h, w, c = ori_img.shape[:3]
            # 1/3
            one_third_img = cv2.resize(ori_img, (h//3, w//3))
            # 1/4
            quarter_img = cv2.resize(ori_img, (h//4, w//4))

            # save image
            cv2.imwrite(os.path.join(one_third_folder, filename), one_third_img)
            cv2.imwrite(os.path.join(quarter_folder, filename), quarter_img)


if __name__ == "__main__":
     # リサイズ
    ROOT = "../ARC_DATAS"
    ARC_DATAS_CROP = "../ARC_DATAS_CROP"
    folderlist = glob.glob(ROOT + "/arcKG*")

    for folder in tqdm(folderlist):
        foldername = os.path.basename(folder)
        arc_crop_folder = os.path.join(ARC_DATAS_CROP, foldername)
        # 画像のフォルダ作成
        os.makedirs(arc_crop_folder, exist_ok=True)
        pathlist = sorted(glob.glob(folder + "/*.jpg"))
        for path in pathlist:    
            filename = os.path.basename(path)
            ori_img = cv2.imread(path)
            h, w, c = ori_img.shape[:3]
            c_x = w//2
            c_y = h // 2
            crop_img = ori_img[c_y: c_y + 256, c_x: c_x + 256]

            # save image
            cv2.imwrite(os.path.join(arc_crop_folder, filename), crop_img)
    