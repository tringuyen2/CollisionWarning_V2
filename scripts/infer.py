from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config

import torch
from torch.utils.data import DataLoader

sys.path.append('.')
sys.path.append('..')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset

import time

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

MIN_DEPTH=1e-3
MAX_DEPTH=100
SCALE = 36#we set baseline=0.0015m which is 36 times smaller than the actual value (0.54m)

def transform(cv2_img, height=320, width=1024):
    # im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cuda().unsqueeze(0)
    im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cpu().unsqueeze(0)

    im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
    im_tensor = torch.nn.functional.interpolate(im_tensor, [height, width],mode='bilinear', align_corners=False)
    im_tensor /= 255
    return im_tensor

def predict(cv2_img, model):
    original_height, original_width = cv2_img.shape[:2]
    im_tensor = transform(cv2_img)

    with torch.no_grad():
        input = {}
        input['color_aug', 0, 0] = im_tensor
        outputs = model(input)

    disp = outputs[("disp", 0, 0)]
    disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
    min_disp = 1/MAX_DEPTH
    max_disp = 1/MIN_DEPTH
    # depth = 1/(disp_resized.squeeze().cpu().numpy()*max_disp + min_disp) * SCALE
    depth = disp_resized.squeeze().cpu().numpy()*100
    depth = 204.3292/(depth) - 1.0848

    return depth, disp_resized.squeeze().cpu().numpy()

def evaluate(cfg_path, model_path, img_path, output_path, file_depth):
    if torch.cuda.is_available():
            #device = torch.device("cuda")
            device = "cuda"
    else:
            device = "cpu"


    cfg = Config.fromfile(cfg_path)
    cfg['model']['depth_pretrained_path'] = None
    cfg['model']['pose_pretrained_path'] = None
    cfg['model']['extractor_pretrained_path'] = None
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # model.cuda()
    model.to(device)
    model.eval()

    with torch.no_grad():
        cv2_img = cv2.imread(img_path)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        t_start = time.time()
        depth, disp_resized = predict(cv2_img, model)
        # print(depth)
        f = open(file_depth, 'w')
        for t in depth:
            line = ' '.join(str(x) for x in t)
            f.write(line + '\n')
        f.close()
        # f = open('file1.txt', 'w')
        # for t in disp_resized:
        #     line = ' '.join(str(x) for x in t)
        #     f.write(line + '\n')
        # f.close()
        print(f"Time 1 image: {time.time() - t_start}")

        vmax = np.percentile(disp_resized, 95)
        plt.imsave(output_path, disp_resized, cmap='magma', vmax=vmax)

    print("\n-> Done!")


if __name__ == "__main__":
    file_depth = '../assets/31.txt'
    # cfg_path = '../config/cfg_kitti_fm.py'# path to cfg file
    cfg_path = '../config/cfg_kitti_fm.py'
    # model_path = '/media/sconly/harddisk/weight/fm_depth.pth'# path to model weight
    model_path = '../epoch_20.pth'
    img_path = '../image-distance/31.png'
    output_path = '../image-distance/31-out.png' # dir for saving depth maps
    evaluate(cfg_path, model_path, img_path, output_path, file_depth)