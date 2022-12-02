from __future__ import absolute_import, division, print_function
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from numpy import random
from random import randint
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.datasets import letterbox

#For SORT tracking
import skimage
from Sort.sort import *

# For GCNNet
import sys
# sys.path.append('.')
# sys.path.append('..')
from mmcv import Config
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

def transform(cv2_img, height=320, width=1024):
    # im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cuda().unsqueeze(0)
    im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cpu().unsqueeze(0)

    im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
    im_tensor = torch.nn.functional.interpolate(im_tensor, [height, width],mode='bilinear', align_corners=False)
    im_tensor /= 255
    return im_tensor


def predict(cv2_img, model):
    MIN_DEPTH=1e-3
    MAX_DEPTH=100
    SCALE = 36#we set baseline=0.0015m which is 36 times smaller than the actual value (0.54m)
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

def distance_estimation(cfg_path, model_path, img_path):
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
        print(f"Time 1 image: {time.time() - t_start}")

        # vmax = np.percentile(disp_resized, 95)
        # plt.imsave(output_path, disp_resized, cmap='magma', vmax=vmax)

    print("\n-> Done!")
    return depth


def detect(
    source, 
    model,
    stride, 
    device,
    imgsz=320,
    augment=False,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic_nms=False
    ):
    img = letterbox(source, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    # pred = non_max_suppression(pred)
    t3 = time_synchronized()
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # print(det)

            points = []
            for box in det:
            
                c1 = (int(box[0]), int(box[1]))
                c2 = (int(box[2]), int(box[3]))
                x1, y1 = c1
                x2, y2 = c2
                acc = round(float(box[4]),2)
                cls = int(box[5])
                points.append([x1, y1, x2, y2, acc, cls])

            # print(points)
            return points



def int_0(a, w, h):
    x1, y1, x2, y2 = a
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h
    return x1, y1, x2, y2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='E:/Data/colision-warning/test1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)

    #============================
    # Load model yolov7
    #============================
    # Initialize
    t_detect = time.time()
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    imgsz = 320
    model_detect = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model_detect.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # model = TracedModel(model, device, imgsz)


    # Get names and colors
    names = model_detect.module.names if hasattr(model_detect, 'module') else model_detect.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if half:
        model_detect.half()  # to FP16
    
    print(f"Time load model detect: {time.time() - t_detect}")
    #=============================
    # Load model gcndepth
    #=============================
    t_depth = time.time()
    cfg_path = './config/cfg_kitti_fm.py'
    model_path = './weights/epoch_20.pth'


    cfg = Config.fromfile(cfg_path)
    cfg['model']['depth_pretrained_path'] = None
    cfg['model']['pose_pretrained_path'] = None
    cfg['model']['extractor_pretrained_path'] = None
    model_depth = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model_depth.load_state_dict(checkpoint['state_dict'], strict=True)
    # model.cuda()
    model_depth.to(device)
    model_depth.eval()

    print(f"Time load model depth estimation: {time.time() - t_depth}")

    # with torch.no_grad():
    #     cv2_img = cv2.imread(img_path)
    #     cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    #     t_start = time.time()
    #     depth, disp_resized = predict(cv2_img, model)
    #     print(depth[0])
    #     print(f"Time 1 image: {time.time() - t_start}")


    # Process video + tracking
    mot_tracker = Sort()

    video_path = 'video-test/1.mp4'

    name = video_path.split('/')[-1]

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)


    size = (frame_width, frame_height)
    result = cv2.VideoWriter(f'result_video/{name[:-4]}.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    f = 0


    while (cap.isOpened()):
        t_start = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cv2.imwrite("frame.jpg", frame)


        h, w = frame_height, frame_width

        frame1 = frame.copy()
        f += 1
        print("###########")
        print(f)

        with torch.no_grad():
            t_detect = time.time()
            dets = detect(frame, model_detect, stride, device, imgsz)
            print(f"Time detect: {time.time() - t_detect}")
            t_depth = time.time()
            depth, disp_resized = predict(frame, model_depth)
            print(f"Time depth: {time.time() - t_depth}")

        

        for i in dets:
            x1, y1, x2, y2 = list(map(int,i[:4]))

        dets = np.array(dets)
        if not len(dets):
            continue
                
        # xywhs = torch.from_numpy(xyxy2xywh(dets[:, 0:4]))
        # confs = torch.from_numpy(dets[:, 4])
        # clss = torch.from_numpy(dets[:, 5])
        # dts = torch.from_numpy(dets[:, :4])
        
        
        trackers = mot_tracker.update(dets)

        
        for i in range(len(trackers)):
            ids = str(int(trackers[i][4]))       
            # print(ids)
            x1, y1, x2, y2 = list(map(int,trackers[i][:4]))
            x1, y1, x2, y2 = int_0([x1, y1, x2, y2], w, h)
            img_crop = frame1[y1:y2, x1:x2]
            distance = np.median(np.array(img_crop))


            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, str(distance) + "m", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)



        t_end = time.time()
        print(f"Time 1 frame: {t_end - t_start}")
        fps = round(1/(t_end-t_start), 1)

        print(f"fps: {fps}")
        print()


        cv2.putText(frame, str(fps) + " fps", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
        
        # Display the resulting frame
        cv2.imshow('Frame', frame)
        result.write(frame)

        # define q as the exit button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
