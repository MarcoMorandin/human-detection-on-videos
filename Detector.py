import cv2
import numpy as np
from utils.motion_detection_utils import *


class HumanDetector:
    def __init__(self, backgound_frame=None):
        # Preprocess the background frame and store it
        self.ref_frame_bg = preprocess_frames(backgound_frame)
        self.frames_dir = "frames"
        self.preprocess_frames_dir = "preprocessed-frames"
        
        # Initialize HOG descriptor for human detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_humans(self, frame, bounding_boxes_optical_flow, overlap_boxes_treshold=0.7):
        # Preprocess the current frame
        frame_preprocessed = preprocess_frames(frame)

        # Process frame differences and find contours
        contours = self._process_frame_differences(frame_preprocessed)
        
        # Get bounding boxes
        bounding_boxes = self._get_bounding_boxes(contours, frame_preprocessed, bounding_boxes_optical_flow, overlap_boxes_treshold=overlap_boxes_treshold)
        
        # Resize the frame for display
        frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)
        
        # Draw bounding boxes on the frame
        for x, y, w, h in bounding_boxes:
            frame = cv2.rectangle(frame, (x , y), (w, h), (0, 0, 200), 2)
            
        return frame, bounding_boxes
    
    def _process_frame_differences(self, frame):
        # Compute the difference between the background frame and the current frame
        mask = cv2.subtract(self.ref_frame_bg, frame)
        
        # Threshold the mask to get binary image
        _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        
        # Apply some filters in order to remove noise and augment the contours of blobs
        thresh = cv2.medianBlur(thresh, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8), iterations=1)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        return contours

    def _get_bounding_boxes(self, contours, frame, bounding_boxes_optical_flow, overlap_boxes_treshold):
        bounding_boxes_dimension = []
        bounding_boxes_merged = []
        bounding_boxes = []
        
        # Get bounding boxes from contours
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if is_valid_box(w, h):
                bounding_boxes_dimension.append([x, y, w+x, h+y])

        # Merge bounding boxes from contours and from optical flow
        for box in bounding_boxes_dimension:
            bounding_boxes_merged.append(box)

        for box in bounding_boxes_optical_flow:
            bounding_boxes_merged.append(box)     

        # Merge bounding boxes without overlap
        merge_without_overlap_boxes=merge_bounding_boxes_while_loop(bounding_boxes_merged, treshold=overlap_boxes_treshold)
       
        # Validate with Hog descriptor each bounding box and filter the non-human detections
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