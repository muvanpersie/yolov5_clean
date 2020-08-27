import argparse
import os
# import platform
# import shutil
import time
# from pathlib import Path

import math 
import glob
import cv2
import torch
# import torch.backends.cudnn as cudnn
import numpy as np


from utils.datasets import letterbox
from utils.general import (non_max_suppression, 
            scale_coords, plot_one_box,  set_logging)
from utils.torch_utils import time_synchronized


def detect(save_img=False):
    source, weights, imgsz = opt.source, opt.weights,  opt.img_size

    img_root_path = "/home/lance/data/DataSets/quanzhou/cyclist/moto/JPEGImages"

    # Initialize
    set_logging()
    device = torch.device('cuda:0')

    # Load model
    model = torch.load(weights[0], map_location=device)['model'].float().fuse().eval()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    max_stride = int(model.stride.max())
    imgsz = math.ceil(opt.img_size / max_stride) * max_stride

    # warm up
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(img)
    
    t0 = time.time()
    img_lists = sorted(glob.glob(img_root_path + '/*.jpg'))
    for img_path in img_lists:
        
        img0 = cv2.imread(img_path)
        # Padded resize
        img = letterbox(img0, new_shape=imgsz)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        # pre.shape --> 1*N*(num_cls+1+4), 其中N为 w/(8,16,32) * h/(8,16,32)总和
        pred = model(img)[0]
        # pred [x1, y1, x2, y2, score, cls]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        print(" Infer time: {:.3f}".format(t2-t1))

        # Process detections
        for i, det in enumerate(pred):  
            
            # normalization gain whwh
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # draw results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

            cv2.imshow("test", img0)
            if cv2.waitKey(0) == ord('q'):  # q to quit
                raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
