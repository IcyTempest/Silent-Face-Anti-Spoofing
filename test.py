# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import shutil

import cv2
import numpy as np
import argparse
import warnings
import time
from pathlib import  Path

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/test_data/live/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def to_3x4(image):
    h, w = image.shape[:2]
    target = 3/4
    if w/h > target:
        new_w = round(h*target)
        x0= (w - new_w) // 2
        return image[:, x0:x0 + new_w]
    else:
        new_h = round(w/target)
        y0 = (h-new_h) // 2
        return image[y0:y0 + new_h, :]

def test(image_name:Path, model_dir, model_test, image_cropper):
    print(str(image_name))
    image = cv2.imread(str(image_name))
    if image is None:
        # raise FileNotFoundError("File Not found")
        print(f"File Not found, {str(image_name)}")
        return

    result = check_image(image)
    if result is False:
        image = to_3x4(image)
        cv2.imwrite(f"images/cropped/{image_name.stem}.jpg", image)

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    img = None
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    result = None
    if label == 1:
        print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
        result = 1
    else:
        print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
        cv2.imwrite(f"images/fake/{image_name.stem}.jpg", image)
        cv2.imwrite(f"images/fake/{image_name.stem}_cropped.jpg", img)
        result = 0

    print("Prediction cost {:.2f} s".format(test_speed))
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    format_ = os.path.splitext(image_name)[-1]

    result_image_name = str(image_name).replace(format_, f"_result{format_}")
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)
    return result

if __name__ == "__main__":
    desc = "test"
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    shutil.rmtree("images/cropped", ignore_errors = True)
    os.makedirs("images/cropped", exist_ok = True)
    shutil.rmtree("images/fake", ignore_errors=True)
    os.makedirs("images/fake", exist_ok=True)

    true = 0
    false = 0
    for sub in os.listdir("real_world_data_processed"):
        for i in list(Path(f"real_world_data_processed/{sub}/live").glob("*")):
            tmp = test(i, "./resources/anti_spoof_models", model_test, image_cropper)
            if tmp:
                true +=1
            else:
                false +=1
            # test(args.image_name, args.model_dir, args.device_id)

    print(f"TP: {true} FP: {false}")
