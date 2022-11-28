import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
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


#............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

"""" Calculates the relative bounding box from absolute pixel values. """
def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""
def draw_boxes(img, bbox, identities=None, categories=None, names=None,offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id) + ":"+ names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)
    return img
#..............................................................................


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

    #.... Initialize SORT .... 
    #......................... 
    # sort_max_age = 5 
    # sort_min_hits = 2
    # sort_iou_thresh = 0.2
    # sort_tracker = Sort(max_age=sort_max_age,
    #                    min_hits=sort_min_hits,
    #                    iou_threshold=sort_iou_thresh) 
    #......................... 
    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    # if trace:
    #     model = TracedModel(model, device, opt.img_size)



    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    # vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    # else:
        # dataset = LoadImages(source, img_size=imgsz, stride=stride)

    

    # # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    # old_img_w = old_img_h = imgsz
    # old_img_b = 1

    t0 = time.time()

    # #........Rand Color for every trk.......
    # rand_color_list = []
    # for i in range(0,5005):
    #     r = randint(0, 255)
    #     g = randint(0, 255)
    #     b = randint(0, 255)
    #     rand_color = (r, g, b)
    #     rand_color_list.append(rand_color)
    #.........................
    img = letterbox(source, imgsz, stride=stride)[0]

    # image = source.copy()
   

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # # Warmup
    # if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
    #     old_img_b = img.shape[0]
    #     old_img_h = img.shape[2]
    #     old_img_w = img.shape[3]
    #     for i in range(3):
    #         model(img, augment=opt.augment)[0]

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)
    t3 = time_synchronized()

    # Apply Classifier
    # if classify:
    #     pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # if webcam:  # batch_size >= 1
        #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        # else:
        #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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

            # image = cv2.rectangle(image, c1, c2, (255, 0, 0), 2)
            # img_crop = source[y1:y2, x1:x2]
            # cv2.imwrite(f"img_crop{x}.jpg", img_crop)
            # cont.append(img_crop)

    #         #..................USE TRACK FUNCTION....................
    #         #pass an empty array to sort
    #         dets_to_sort = np.empty((0,6))
            
    #         # NOTE: We send in detected object class too
    #         for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
    #             dets_to_sort = np.vstack((dets_to_sort, 
    #                         np.array([x1, y1, x2, y2, conf, detclass])))
            
    #         # Run SORT
    #         tracked_dets = sort_tracker.update(dets_to_sort)
    #         tracks =sort_tracker.getTrackers()
            
    #         #loop over tracks
    #         for track in tracks:
    #             # color = compute_color_for_labels(id)
    #             #draw tracks
    #             [cv2.line(source, (int(track.centroidarr[i][0]),
    #                             int(track.centroidarr[i][1])), 
    #                             (int(track.centroidarr[i+1][0]),
    #                             int(track.centroidarr[i+1][1])),
    #                             rand_color_list[track.id], thickness=2) 
    #                             for i,_ in  enumerate(track.centroidarr) 
    #                                 if i < len(track.centroidarr)-1 ] 
    #         # draw boxes for visualization
    #         if len(tracked_dets)>0:
    #             bbox_xyxy = tracked_dets[:,:4]
    #             identities = tracked_dets[:, 8]
    #             categories = tracked_dets[:, 4]
    #             draw_boxes(im0, bbox_xyxy, identities, categories, names)
    #         #........................................................
            
    #     # Print time (inference + NMS)
    #     print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
    #     cv2.putText(im0, str(round(1/(time.time() - t1),1)) + " fps", (50, 50), 0, 1, (255, 128, 0), 1)

    #     # Stream results
    #     if view_img:
    #         cv2.imshow(str(p), im0)
    #         cv2.waitKey(1)  # 1 millisecond

    #     # Save results (image with detections)
    #     if save_img:
    #         if dataset.mode == 'image':
    #             cv2.imwrite(save_path, im0)
    #             print(f" The image with the result is saved in: {save_path}")
    #         else:  # 'video' or 'stream'
    #             if vid_path != save_path:  # new video
    #                 vid_path = save_path
    #                 if isinstance(vid_writer, cv2.VideoWriter):
    #                     vid_writer.release()  # release previous video writer
    #                 if vid_cap:  # video
    #                     fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #                     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #                     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #                 else:  # stream
    #                     fps, w, h = 30, im0.shape[1], im0.shape[0]
    #                     save_path += '.mp4'
    #                 vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #             vid_writer.write(im0)

    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     #print(f"Results saved to {save_dir}{s}")

    # print(f'Done. ({time.time() - t0:.3f}s)')


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
    #check_requirements(exclude=('pycocotools', 'thop'))

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    imgsz = 320
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if half:
        model.half()  # to FP16
    
    # img = cv2.imread('image-test/test1.png')
    # points = detect(img, model, stride, device, imgsz)


    # Process video + tracking
    mot_tracker = Sort()

    video_path = 'video-test/test2.mp4'

    name = video_path.split('/')[-1]

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)


    size = (frame_width, frame_height)
    result = cv2.VideoWriter(f'result_video/{name[:-4]}.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    # if not os.path.exists(f'results/{name[:-4]}'):
    #     os.mkdir(f'results/{name[:-4]}')

    f = 0
    count_id = dict()
    # Loop until the end of the video
    while (cap.isOpened()):
        keys = count_id.keys()
        t_start = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()


        h, w = frame_height, frame_width

        frame1 = frame.copy()
        f += 1
        print("###########")
        print(f)

        t_start = time.time()
        # dets = detect_lp(frame, mods_lp, device=device, imgsz=imgsz)
        dets = detect(frame, model, stride, device, imgsz)
        t_end = time.time()
        print(f"Time detect lp: {t_end - t_start}")
        

        for i in dets:
            x1, y1, x2, y2 = list(map(int,i[:4]))
        dets = np.array(dets)
        if not len(dets):
            continue
                
        xywhs = torch.from_numpy(xyxy2xywh(dets[:, 0:4]))
        confs = torch.from_numpy(dets[:, 4])
        clss = torch.from_numpy(dets[:, 5])
        dts = torch.from_numpy(dets[:, :4])
        
        
        trackers = mot_tracker.update(dets)

        
        for i in range(len(trackers)):
            ids = str(int(trackers[i][4]))       
            # print(ids)
            x1, y1, x2, y2 = list(map(int,trackers[i][:4]))
            x1, y1, x2, y2 = int_0([x1, y1, x2, y2], w, h)
            img_crop = frame1[y1:y2, x1:x2]


            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, str(ids), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,125,0), 1)



        t_end = time.time()
        print(f"Time 1 frame: {t_end - t_start}")
        fps = round(1/(t_end-t_start), 1)

        print(f"fps: {fps}")
        print()


        cv2.putText(frame, str(fps) + " fps", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)
        
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        result.write(frame)

        # print(f"count_id: {count_id}")

        # define q as the exit button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
