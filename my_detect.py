import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



class ObjectDetector:
    def __init__(self, weights='HighAlert_custom_weights.pt', img_size=640, conf_thres=0.4, iou_thres=0.45,
                 device='', augment=False, no_trace=True):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.augment = augment
        self.no_trace = no_trace
        
        # Load model
        set_logging()
        self.device = select_device(self.device)
        self.model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size

        if self.no_trace:
            self.model = TracedModel(self.model, self.device, self.img_size)
            
        self.half = self.device.type != 'cpu' and self.augment  # half precision only supported on CUDA

        if self.half:
            self.model.half()  # to FP16
            
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz , self.imgsz ).to(self.device).type_as(next(self.model.parameters())))  # run once
            
            
    


    def scale_coords(self, img0_shape, coords, img1_shape):
        # Ensure input shapes are not empty tuples
        if len(img0_shape) < 2 or len(img1_shape) < 2:
            return coords

        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords
    
    def clip_coords(self, coords, img_shape):
        coords[:, 0].clamp_(0, img_shape[1])  # x1
        coords[:, 1].clamp_(0, img_shape[0])  # y1
        coords[:, 2].clamp_(0, img_shape[1])  # x2
        coords[:, 3].clamp_(0, img_shape[0])  # y2



    def detect(self, frame):
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("Error: Invalid frame")
            return []
        

            
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV uses BGR by default)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img[np.newaxis, ...] / 255.0  # Add batch dimension, Normalize

        # Convert NumPy array to PyTorch tensor and move to device
        img = torch.from_numpy(img).float().to(self.device)
        
        img = img.half() if self.half else img.float()  # uint8 to fp16/32


        img = self.model(img, augment=True)[0]
        pred = non_max_suppression(img, self.conf_thres, self.iou_thres)
        
        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(frame.shape, det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xyxy = [int(xy) for xy in xyxy]
                    detections.append({'bbox': xyxy, 'confidence': conf.item(), 'class': int(cls)})
        return detections



def main():
    cap = cv2.VideoCapture(0)  # Open camera

    object_detector = ObjectDetector()  # Initialize object detector

    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame

        if not ret:
            print("Error: Couldn't capture frame.")
            break

        # Detect objects in the frame
        detections = object_detector.detect_frame(frame)

        # Draw bounding boxes and labels
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class']
            confidence = detection['confidence']
            label = f'Class: {class_id}, Confidence: {confidence:.2f}'
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()