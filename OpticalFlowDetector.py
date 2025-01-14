import cv2
import numpy as np
from utils.motion_detection_utils import *

class OpticalFlowTracking():

    def __init__(self):
        """
        Initialize the class with default settings for frame dimensions, morphological kernel,
        and the background subtractor.
        """
        # Standard frame dimentions for resizing
        self.height = 768
        self.width = 1366
        # Morpohology kernel for image processing
        self.kernel = np.ones((5, 5), dtype=np.uint8)
        # Initialize the background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=True
        )

    def detect_humans(self, frame1, frame2, combine_bboxes_thresh=0.15):
        """
        Detect if a frame contains person/people

        Parameters:
            frame1: First frame in a pair of frame
            frame2: Second frame in a pair of frame
            combine_bboxes_thresh: Threshold for merging bounding boxes (IoU-based)

        Returns:
            detections: Detected bounding boxes about the second frame
        """
        # Resize frames to standard dimensions to avoid proportions inconsistency
        frame1 = cv2.resize(frame1, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        detections = self.get_detections(frame1, frame2, motion_thresh=1, combine_bboxes_thresh=combine_bboxes_thresh)
        return detections

    def compute_flow(self, gray1, gray2, 
                    pyr_scale=0.5,    
                    levels=4,        
                    winsize=15,      
                    iterations=3,    
                    poly_n=7,        
                    poly_sigma=1.5,  
                    flow_flags=0):
        
        """
        Compute the dense optical flow between two grayscale frames using the Farneback method

        Parameters:
            gray1: First grayscale frame
            gray2: Second grayscale frame

        Returns:
            flow: Dense optical flow
        """

        # Apply Gaussian blur to reduce noise and improve accuracy
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
        """
        Obtain a motion mask based on the magnitude of optical flow vectors

        Parameters:
            flow_mag: Magnitude of optical flow vectors
            motion_thresh: Threshold for significant motion
            kernel: Morphological kernel for morphology action

        Returns:
            motion_mask: Binary mask highlighting regions with significant motion
        """
        motion_mask = np.uint8(flow_mag > motion_thresh)*255
        # Close the small withe spaces between near areas mantaining the shape of the original reagion
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        return motion_mask

    
    def get_edge_mask(self, gray_frame, low_thresh=100, high_thresh=200):
        """
        Perform Canny edge detection on a grayscale frame. Used to reduce false 
        positives in detections

        Parameters:
            gray_frame: Input grayscale frame
            low_thresh: Lower threshold for edge detection
            high_thresh: Upper threshold for edge detection

        Returns:
            Binary edge mask highlighting edges in the frame
        """
        return cv2.Canny(gray_frame, threshold1=low_thresh, threshold2=high_thresh)
        
    def get_optical_flow_mask(self, frame1, frame2, motion_thresh):
        """
        Generate a binary motion mask using optical flow

        Parameters:
            frame1: First input frame
            frame2: Second input frame
            motion_thresh: Threshold for significant motion

        Returns:
            Binary motion mask based on optical flow magnitude
        """
        flow = self.compute_flow(frame1, frame2)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Filter only relevant movement
        return self.get_motion_mask(mag, motion_thresh, self.kernel)

    def get_background_subtraction_mask(self, frame):
        """
        Generate a foreground mask using background subtraction

        Parameters:
            frame: Input frame

        Returns:
            fg_mask: Foreground mask highlighting moving regions
            gray_frame: Grayscale version of the input frame
        """
        fg_mask = self.bg_subtractor.apply(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return fg_mask, gray_frame


    def get_final_mask(self, optical_flow_mask, fg_mask, edge_mask):
        """
        Combine motion, background subtraction, and edge masks into a final detection mask

        Parameters:
            optical_flow_mask: Binary mask from optical flow
            fg_mask: Foreground mask from background subtraction
            edge_mask: Edge mask from Canny edge detection

        Returns:
            Final binary mask combining all input masks
        """
        combined_mask = cv2.bitwise_and(optical_flow_mask, fg_mask)
        combined_mask = cv2.bitwise_and(combined_mask, edge_mask)

        # Morphological cleanup on the final mask
        # Used to links neighbouring areas of white pixels.
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, self.kernel, iterations=1)
        # Convert to binary for contour detection
        return np.uint8(combined_mask > 0) * 255


    def check_validity(self, detections):
        """
        Validate the detected bounding boxes to ensure the proportions of a person is respected

        Parameters:
            detections: List of detected bounding boxes

        Returns:
            Filtered list of valid detections
        """
        order = list(range(0, len(detections)))
        keep = []
        while order:
            i = order.pop(0)
            if is_valid_box(detections[i][2], detections[i][3]):
                keep.append(i)
        return detections[keep]
        
    def get_detections(self, frame1, frame2,
                       motion_thresh=1,
                       combine_bboxes_thresh=0.1):
        
        """
        Detect person/people and generate bounding boxes around them

        Parameters:
            frame1: First input frame
            frame2: Second input frame
            motion_thresh: Threshold for significant motion
            combine_bboxes_thresh: IoU threshold for merging bounding boxes

        Returns:
            Bounding boxes around detected objects
        """
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
        detections = get_contour_detections(final_mask)
        if len(detections) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        bboxes = detections[:, :4]
        bboxes = merge_bounding_boxes_while_loop(bboxes,combine_bboxes_thresh, check_validity=True)

        return self.check_validity(bboxes)