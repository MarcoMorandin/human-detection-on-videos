import cv2
import os
import numpy as np
from tqdm import tqdm
from utils.motion_detection_utils import *


class HumanDetector:
    def __init__(self, backgound_frame=None):
        self.ref_frame_bg = self._preprocess_frames(backgound_frame)
        self.frames_dir = "frames"
        self.preprocess_frames_dir = "preprocessed-frames"
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_humans(self, frame, bounding_boxes_optical_flow, overlap_boxes_treshold=0.7):

        frame_preprocessed = self._preprocess_frames(frame)

        contours = self._process_frame_differences(frame_preprocessed)
        bounding_boxes = self._get_bounding_boxes(contours, frame_preprocessed, bounding_boxes_optical_flow, overlap_boxes_treshold=overlap_boxes_treshold)
        
        frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)
        
        for x, y, w, h in bounding_boxes:
            frame = cv2.rectangle(frame, (x , y), (w, h), (0, 0, 200), 2)
        return frame, bounding_boxes
    
    def _process_frame_differences(self, frame):
        mask = cv2.subtract(self.ref_frame_bg, frame)
        _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        
        thresh = cv2.medianBlur(thresh, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        return contours

    def _get_bounding_boxes(self, contours, frame, bounding_boxes_optical_flow, overlap_boxes_treshold):
        bounding_boxes_dimension = []
        bounding_boxes_merged = []
        bounding_boxes = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if self._is_valid_box(w, h):
                bounding_boxes_dimension.append([x, y, w+x, h+y])

        for box in bounding_boxes_dimension:
            bounding_boxes_merged.append(box)

        for box in bounding_boxes_optical_flow:
            if self._is_valid_box(box[2], box[3]):
                bounding_boxes_merged.append(box)     

        merge_without_overlap_boxes=merge_bounding_boxes_while_loop(bounding_boxes_merged, treshold=overlap_boxes_treshold)
       
        for x, y, w, h in merge_without_overlap_boxes:
            crop = frame[max(0, y - 50):h + 50, max(0, x - 50):w + 50]
            if crop.shape[0] > 0 and crop.shape[1] > 0:    
                if len(merge_without_overlap_boxes) < 10:  # Limit the number of detections in order to avoid fulling memory else appened without svm
                    humans = self.hog.detectMultiScale(crop)
                    if len(humans) > 0:
                        bounding_boxes.append((x, y, w, h))
                else:
                    bounding_boxes.append((x, y, w, h))
        return bounding_boxes
        

    @staticmethod
    def _preprocess_frames(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)
        return frame
    
    
    @staticmethod
    def _is_valid_box(width, height):
        area = width * height
        aspect_ratio = float(width) / height
        return 0.2 < aspect_ratio < 1 and area > 1500
    