import cv2
import numpy as np
from utils.motion_detection_utils import *

class OpticalFlowTracking():

    def __init__(self):
        self.height = 768
        self.width = 1366
        self.kernel = np.ones((5, 5), dtype=np.uint8)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=True
        )

    def detect_humans(self, frame1, frame2, combine_bboxes_thresh=0.25):
        frame1 = cv2.resize(frame1, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        detections = self.get_detections(frame1, frame2, motion_thresh=1, bbox_thresh=400, combine_bboxes_thresh=combine_bboxes_thresh)
        return detections

    def compute_flow(self, gray1, gray2, 
                    pyr_scale=0.5,    
                    levels=4,        
                    winsize=15,      
                    iterations=3,    
                    poly_n=7,        
                    poly_sigma=1.5,  
                    flow_flags=0):
        
        """Compute the optical flow between two grayscale frames."""
        gray1 = cv2.GaussianBlur(gray1, dst=None, ksize=(3,3), sigmaX=5)
        gray2 = cv2.GaussianBlur(gray2, dst=None, ksize=(3,3), sigmaX=5)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=flow_flags
        )
        return flow

    def get_motion_mask(self, flow_mag, motion_thresh=1, kernel=np.ones((9,), np.uint8)):
        """ Obtains Detection Mask from Optical Flow Magnitude
            Inputs:
                flow_mag (array) Optical Flow magnitude
                motion_thresh - thresold to determine motion
                kernel - kernal for Morphological Operations
            Outputs:
                motion_mask - Binray Motion Mask
            """
        motion_mask = np.uint8(flow_mag > motion_thresh)*255
        # close the small withe spaces between near areas mantaining the shape of the original reagion
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        return motion_mask

    # This can reduce false positives in homogeneous areas by focusing on structural features
    def get_edge_mask(self, gray_frame, low_thresh=100, high_thresh=200):
        """Perform Canny edge detection and return a binary edge mask."""
        return cv2.Canny(gray_frame, threshold1=low_thresh, threshold2=high_thresh)
        
    def get_optical_flow_mask(self, frame1, frame2, motion_thresh):
        flow = self.compute_flow(frame1, frame2)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # filter only relevant movement
        return self.get_motion_mask(mag, motion_thresh, self.kernel)

    def get_background_subtraction_mask(self, frame):
        """Get the foreground mask from the background subtractor."""
        fg_mask = self.bg_subtractor.apply(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return fg_mask, gray_frame


    def get_final_mask(self, optical_flow_mask, fg_mask, edge_mask):
        """Combine the motion, background subtraction, and edge masks."""
        #   1) motion_mask  => captures movement from optical flow
        #   2) fg_mask      => highlights moving objects vs. the background
        #   3) edge_mask    => focuses on structural boundaries
        combined_mask = cv2.bitwise_and(optical_flow_mask, fg_mask)
        combined_mask = cv2.bitwise_and(combined_mask, edge_mask)

        # Morphological cleanup on the final mask
        #MORPH_DILATE: used to links neighbouring areas of white pixels.
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, self.kernel, iterations=1)
        # Convert to binary for contour detection
        return np.uint8(combined_mask > 0) * 255


    def check_validity(self, detections):
        order = list(range(0, len(detections)))
        keep = []
        while order:
            i = order.pop(0)
            if check(detections[i][0], detections[i][1], detections[i][2], detections[i][3]):
                keep.append(i)
        return detections[keep]
        
    def get_detections(self, frame1, frame2,
                       motion_thresh=1,
                       bbox_thresh=400,
                       combine_bboxes_thresh=0.1):
        """Main function to get detections using combined masks."""
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute optical flow and motion mask
        optical_flow_mask=self.get_optical_flow_mask(gray1, gray2, motion_thresh)

        # Background subtraction
        fg_mask, gray_frame2 = self.get_background_subtraction_mask(frame2)

        #edge detection
        edge_mask = self.get_edge_mask(gray_frame2, low_thresh=50, high_thresh=150)

        # Combined mask
        final_mask = self.get_final_mask(optical_flow_mask, fg_mask, edge_mask)

        # Detect contours and bounding boxes
        detections = get_contour_detections(final_mask, thresh=bbox_thresh)
        if len(detections) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        bboxes = detections[:, :4]
        bboxes = merge_bounding_boxes_while_loop(bboxes, check_validity=True, treshold=combine_bboxes_thresh)
        return self.check_validity(bboxes)