import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, 
            scale_coords, plot_one_box,  set_logging)
from utils.torch_utils import time_synchronized


def detect(save_img=False):
    source, weights, imgsz = opt.source, opt.weights,  opt.img_size

    # Initialize
    set_logging()
    device = torch.device('cuda:0')

    # Load model
    model = torch.load(weights[0], map_location=device)['model'].float().fuse().eval()
    # model.half()  # to FP16

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    imgsz = check_img_size(imgsz, s=model.stride.max())
    dataset = LoadImages(source, img_size=imgsz)

    # warm up
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    # _ = model(img.half())
    _ = model(img)
    
    t0 = time.time()
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]
        # pre.shape 1*N*(num_cls+1+4), 其中N为 w/(8,16,32) * h/(8,16,32)总和

        t2 = time_synchronized()

        # pred [x1, y1, x2, y2, score, cls]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        
        print ("Image size: ", img.shape[2:])
        print(" Infer time: {:.3f}".format(t2-t1))

        # Process detections
        for i, det in enumerate(pred):  
            
            # normalization gain whwh
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # draw results
                for *xyxy, conf, cls in reversed(det):
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)

            cv2.imshow("test", im0s)
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
