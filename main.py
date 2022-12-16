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

from collections import deque



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
    # depth = 204.3292/(depth) - 1.0848
    depth = 0.0036* np.square(depth) - 0.5373 * depth + 21.714


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
    parser.add_argument('--weights_detect', nargs='+', type=str, default='./weights/yolov7-tiny.pt', help='model yolov7 path(s)')
    parser.add_argument('--weights_depth', nargs='+', type=str, default='./weights/epoch_20.pth', help='model gcndepth path(s)')
    parser.add_argument('--source', type=str, default='E:/Data/colision-warning/test1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
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
    model_detect = attempt_load(opt.weights_detect, map_location=device)  # load FP32 model
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
    model_path = opt.weights_depth


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


        
    #========================
    # Initialize video config
    #========================

    # Process video + tracking
    mot_tracker = Sort()

    video_path = 'video-test/demo4_30.mp4'

    name = video_path.split('/')[-1]

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)


    size = (frame_width, frame_height)
    # result = cv2.VideoWriter(f'result_video/{name[:-4]}.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    result = cv2.VideoWriter(f'result_video/{name[:-4]}_2.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)



    #=============================
    #Initialize collision warning
    #=============================
    # Lấy tiêu cự
    # with open('focalLength.txt') as f:
    with open('focalLength-phone.txt') as f:
        focal_length = float(f.read())

    # Khởi tạo chiều cao trung bình của xe để ước lượng khoảng cách
    real_car_height = 1.6
    real_truck_height = 3.5
    real_bus_height = 3.2
    real_motorbike_height = 1.2
    real_person_height = 1.7

    allowed_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'person']

    
    # Khởi tạo hàng đợi lưu lại điểm trung tâm bounding box để hiển thị tracking
    pts = [deque(maxlen=30) for _ in range(9999)]

    # Khởi tạo hàng đợi chứa khoảng cách từng vật thể đang tracking
    distances = [deque(maxlen=30) for _ in range(9999)]

    # Velocity in video test (km/h)
    v = 25  # km/h
    v = v * 0.6214 # mph

    # Distance warning
    distance_warning = v * 0.671 # met

    # Draw front area
    # Define point in front area
    point1 = (int(frame_width * 0.2), frame_height)
    point3 = (int(frame_width * 0.8), frame_height)
    point2 = (int((point1[0] + point3[0])/2), int(frame_height * 0.6))
    tan_front_area = (frame_height - point2[1]) / ((point3[0] - point1[0])/2)

    f = 0


    while (cap.isOpened()):
        t_start = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cv2.imwrite("frame.jpg", frame)


        h, w = frame_height, frame_width
        print(f"h = {h}, w = {w}")

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
            depth_arr = np.asarray(depth)
            # print(f"depth[5][5]: {depth_arr[5][5]}")
            # print(f"{depth.shape}")
            print(f"Time depth: {time.time() - t_depth}")


        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        

        for i in dets:
            x1, y1, x2, y2, acc, cls = list(map(int,i))

        dets = np.array(dets)
        if not len(dets):
            continue

        trackers = mot_tracker.update(dets)
        warn = 0
        
        for i in range(len(trackers)):
            x1, y1, x2, y2, acc, cls = list(map(int,trackers[i]))
            # x1, y1, x2, y2 = int_0([x1, y1, x2, y2], w, h)

            class_name = str(names[cls])
            if class_name not in allowed_classes:
                continue


            ids = int(trackers[i][4])   
            color = colors[ids % len(colors)]
            color = [i * 255 for i in color] 

            # Draw motion path
            center = ((x1 + x2)//2, (y1 + y2)//2)
            pts[ids].append(center)
            cv2.circle(frame,  (center), 1, color, 5)
            for j in range(1, len(pts[ids])):
                if pts[ids][j - 1] is None or pts[ids][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame, (pts[ids][j-1]), (pts[ids][j]), (color), thickness)
            

            # Caculate and draw distance
            depth_crop = depth_arr[y1:y2,x1:x2]
            # print(depth_crop)
            distance = np.median(depth_crop)
            # distance = depth_arr[(y1+y2)//2, (x1+x2)//2]

            if distance > 10:
                if (class_name == 'car'):
                    distance = (real_car_height * focal_length * frame_height) / (y2 - y1)
                elif (class_name == 'truck'):
                    distance = (real_truck_height * focal_length * frame_height) / (y2 - y1)
                elif (class_name == 'bus'):
                    distance = (real_bus_height * focal_length * frame_height) / (y2 - y1)
                elif (class_name == 'motorbike' or class_name == 'bicycle'):
                    distance = (real_motorbike_height * focal_length * frame_height) / (y2 - y1)
                elif (class_name == 'person'):
                    distance = (real_person_height * focal_length * frame_height) / (y2 - y1)


            a_point = center #Vị trí center bounding box thời điểm hiện tại
            b_point = pts[ids][0]    # Vị trí center bounding box thời điểm 30 frame trước
            # Check and draw orientaiton, = 1 if damage
            if ((a_point[0] > b_point[0]) and (a_point[0] < point2[0])) or ((a_point[0] < b_point[0]) and (a_point[0] > (point1[0] + point3[0])/2)): # Trường hợp xe có hướng vào vùng phía trước xe tiếp tục xét góc đi vào, ngược lại (xe không có hướng đi vào vùng trước) thì an toàn
                tan = (b_point[1] - a_point[1]) / (b_point[0] - a_point[0])
                if tan < 0: tan = -tan
                # cv2.putText(frame, str(round(tan, 2)), (int(
                #         (bbox[0]+bbox[2])/2)-30, int((bbox[1]+bbox[3])/2)-30), 0, 0.75, (255, 255, 255), 2)
                if (a_point[1] > b_point[1]): # Trường hợp xe đang xét xu hướng lùi lại và đi vào giữa
                    if tan < 0.7:
                        orientation = 1                     
                    else:
                        orientation = 0
                        
                else:               
                    if tan < 0.2:   # Trường hợp xe đang xét xu hướng tiến lên và đi vào giữa
                        orientation = 1
                    else:
                        orientation = 0
            else:
                orientation = 0

            # 
            x = 0
            y = 0
            tan_center=0
            area_emergency = 0
            # Kiểm tra xe có ở vùng nguy hiểm không, = 1 nếu trong vùng, = 2 nếu chỉ chạm, = 0 nếu không chạm
            if (center[0] < point1[0]) or (center[0] > point3[0]) or (y2 < point2[1]):# Chia thành 4 vùng trái, giữa, phải, vùng trên trời nếu là trái hoặc phải hoặc trên trời thì ở vùng an toàn-> xét phần ở giữa
                area_emergency = 0
                x = 1
            else: # Xét vùng ở giữa
                # Tính góc vị trí của điểm trung tâm
                if (center[0] < (point1[0] + point3[0])/2) and (center[0] > point1[0]): # Trường hợp bên trái vùng phía trước đang xét
                    tan_center = (frame_height - center[1]) / (center[0] - point1[0]) if (center[0] - point1[0]) != 0 else (frame_height - center[1]) / 0.001
                    x = 2.1
                    # Kiểm tra điểm trung tâm có nằm trong vùng nguy hiểm không
                    if tan_center < tan_front_area:
                        area_emergency = 1 # Nằm trong vùng nguy hiểm
                        y = 2
                    else:
                        # Nếu không thì kiểm tra bounding box có chạm vùng nguy hiểm không
                        
                        tan_box = (frame_height - y2) / (x2 - point1[0]) if x2 - point1[0] != 0 else (frame_height - y2) / 0.001
                        x = 3.1
                        
                        if tan_box < tan_front_area:
                            area_emergency = 2 #Chạm vùng nguy hiểm
                            y = 3.1
                        else:
                            area_emergency = 0
                            y = 3.2
                elif (center[0] > (point1[0] + point3[0])/2) and (center[0] < point3[0]): # Trường hợp bên phải vùng phía trước đang xét
                    tan_center = (frame_height - center[1]) / (point3[0] - center[0]) if point3[0] - center[0] != 0 else (frame_height - center[1]) / 0.001
                    x = 2.2
                    # Kiểm tra điểm trung tâm có nằm trong vùng nguy hiểm không
                    if tan_center < tan_front_area:
                        area_emergency = 1 # Nằm trong vùng nguy hiểm
                        y = 2
                
                    else:
                        # Nếu không thì kiểm tra bounding box có chạm vùng nguy hiểm không
                       
                        tan_box = (frame_height - y2) / (point3[0] - x1) if point3[0] - x1 != 0 else (frame_height - y2) / 0.001
                        x = 3.2
                        if tan_box < tan_front_area:
                            area_emergency = 2
                            y = 3.1
                        else:
                            area_emergency = 0
                            y = 3.2
            

            # Vẽ mũi tên cho xe có hướng nguy hiểm
            if distance < 10 and orientation == 1 and area_emergency != 1:
                if center[0] < ((point1[0] + point3[0])/2):
                    cv2.arrowedLine(frame, (int(x1), int(y1-10)), (int((x1+x2)/2), int(y1-10)), (255, 215, 0), 3)
                else:
                    cv2.arrowedLine(frame, (int(x2), int(y1-10)), (int((x1+x2)/2), int(y1-10)), (255, 215, 0), 3)


            # draw bbox on screen
            if ((area_emergency == 1) and (distance < 0.5*distance_warning)) or (distance < 0.5*distance_warning and orientation == 1): # Những trường hợp nguy cấp
                color = (255, 0, 0)
                warn = 1
            elif (area_emergency == 2 and distance < 0.8 * distance_warning) or (distance < 0.2*distance_warning) or ((area_emergency == 1) and (distance < distance_warning)) or (distance < distance_warning and orientation == 1): # Những trường hợp cảnh báo
                color = (255, 215, 0)
                if warn == 0:
                    warn = 2
            else:   # Những trương hợp an toàn
                color = (0, 0, 255)

            # print(color)
            # cv2.circle(frame,  (center), 1, color, 5)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, str(round(distance,1)) + "m", ((x1 + x2)//2, (y1 + y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
            cv2.putText(frame, str(round(distance, 2)) + str(" m"), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
            # cv2.putText(frame, str(ids), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)



        # Vẽ vùng phía trước để cảnh báo
        cv2.line(frame, point1, point2, (255, 64, 64), 1)
        cv2.line(frame, point3, point2, (255, 64, 64), 1)

        t_end = time.time()
        print(f"Time 1 frame: {t_end - t_start}")
        fps = round(1/(t_end-t_start), 1)

        if warn == 1:
            cv2.putText(frame, "Emergency!!", (50, 100), 0, 1.5, (255, 0, 0), 3)
        elif warn == 2:
            cv2.putText(frame, "Warning!!", (50, 100), 0, 1.5, (255, 215, 0), 3)
        else:
            cv2.putText(frame, "Safe", (50, 100), 0, 1.5, (0, 0, 255), 3)


        print(f"fps: {fps}")
        print()


        cv2.putText(frame, str(fps) + " fps", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
        
        # Display the resulting frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', frame)
        result.write(frame)

        # define q as the exit button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
