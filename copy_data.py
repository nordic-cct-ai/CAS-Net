import os
import numpy as np
import nibabel as nib
import json


def load_json(json_file_name):
    with open(json_file_name) as f:
        d = json.load(f)
    return d

path_to_data = "/home/fredrik/Documents/GitHub/coronary_artery_segmentation/data"
path_to_data_new_location = "/media/fredrik/server_data/tmp_save_data"
json_file = "st_olav_ok_ML_roi_prediction.json"
input_dict = load_json(json_file)

cases_to_copy = ["training", "validation", "test"]


os.makedirs(os.path.join(path_to_data_new_location, "img"), exist_ok=True)
os.makedirs(os.path.join(path_to_data_new_location, "mask"), exist_ok=True)
case_conversion_dict = {"ori_name": [], "new_name": []}
cur_name = 1
for tmp_case in cases_to_copy:
    cur_cases = input_dict[tmp_case]

    for tmp_patient in cur_cases:
        img_file = os.path.join(path_to_data, tmp_patient["image"])
        seg_file = os.path.join(path_to_data, tmp_patient["label"])
        img_file_save_npy = os.path.join(path_to_data_new_location, "img", f"{cur_name}.npy")
        seg_file_save_npy = os.path.join(path_to_data_new_location, "mask", f"{cur_name}.npy")

        # Load the NIfTI image
        img = nib.load(img_file)
        seg_img = nib.load(seg_file)

        # Get the image data as a NumPy array
        img_array = img.get_fdata()
        seg_array = seg_img.get_fdata()

        ori_name = img_file.split("/")[-2]
        case_conversion_dict["ori_name"].append(ori_name)
        case_conversion_dict["new_name"].append(f"{cur_name}")
        print(img_file, img_file_save_npy)
        print(seg_file, seg_file_save_npy)
        print("\n")
        np.save(img_file_save_npy, img_array)
        np.save(seg_file_save_npy, seg_array)
        cur_name += 1

