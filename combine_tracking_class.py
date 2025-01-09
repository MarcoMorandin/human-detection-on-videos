import cv2
from tqdm import tqdm
import glob as gl
import numpy as np
from motion_detection_utils import *

class OpticalFlowTracking():

    def __init__(self):
        self.output_video = "human-detection-optical-flow-combined-class.avi"
        self.frames_dir = "frames"
        self.height = 576
        self.width = 768
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=50,
            detectShadows=True
        )

    def get_bounded_boxes(self):
        self.main_with_optical_flow(self.frames_dir, self.output_video, self.height, self.width)

    def load_grayscale_image(self, frame):
        """Convert a frame to grayscale."""
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error loading image {frame}: {e}")
            return None

    def compute_flow(self, gray1, gray2,
                    pyr_scale=0.5,    # recommended range: [0.3, 0.6]  0.75
                    levels=4,        # recommended range: [3, 6]       3
                    winsize=15,      # recommended range: [5, 21]      5
                    iterations=3,    # recommended range: [3, 10]      3
                    poly_n=7,        # recommended range: [5, 7]       10
                    poly_sigma=1.5,  # recommended range: [1.1, 1.5]   1.2
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
        #motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        # used to eliminate folse positive (small variation in the optical flow)
        # then the  dilation restores the original size of larger objects, but 
        # leaves out the small fragments of noise removed by erosion.
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # MORPH_CLOSE: the goal is to remove the small empty spaces eliminating the smaller
        # disconnections mantaining the shape of the original reagion
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        return motion_mask

    # This can reduce false positives in homogeneous areas by focusing on structural features
    def get_edge_mask(self, gray_frame, low_thresh=100, high_thresh=200):
        """Perform Canny edge detection and return a binary edge mask."""
        edges = cv2.Canny(gray_frame, threshold1=low_thresh, threshold2=high_thresh)
        return edges
    
    def get_optical_flow_mask(self, frame1, frame2, motion_thresh, mask_kernel):
        flow = self.compute_flow(frame1, frame2)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # filter only relevant movement
        return self.get_motion_mask(mag, motion_thresh, mask_kernel)

    def get_background_subtraction_mask(self, frame):
        """Get the foreground mask from the background subtractor."""
        fg_mask = self.bg_subtractor.apply(frame)
        gray_frame = self.load_grayscale_image(frame)
        return fg_mask, gray_frame

    def get_final_mask(self, optical_flow_mask, fg_mask, edge_mask, kernel):
        """Combine the motion, background subtraction, and edge masks."""
                #   1) motion_mask  => captures movement from optical flow
        #   2) fg_mask      => highlights moving objects vs. the background
        #   3) edge_mask    => focuses on structural boundaries
        combined_mask = cv2.bitwise_and(optical_flow_mask, fg_mask)
        combined_mask = cv2.bitwise_and(combined_mask, edge_mask)

        # Morphological cleanup on the final mask
        #MORPH_DILATE: used to links neighbouring areas of white pixels.
        #Rendere più evidenti i bordi rilevati da Canny Edge Detection.
        #Riempire eventuali buchi lasciati dal Background Subtraction.
        #Unire aree vicine di movimento individuate dall'Optical Flow.
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        # Convert to binary for contour detection
        return np.uint8(combined_mask > 0) * 255
        
    def get_detections(self, frame1, frame2,
                       motion_thresh=1,
                       bbox_thresh=400,
                       kernel=np.ones((7, 7), dtype=np.uint8),
                       combine_bboxes_thresh=0):
        """Main function to get detections using combined masks."""
        # Convert frames to grayscale
        gray1 = self.load_grayscale_image(frame1)
        gray2 = self.load_grayscale_image(frame2)

        # Compute optical flow and motion mask
        optical_flow_mask=self.get_optical_flow_mask(gray1, gray2, motion_thresh, kernel)

        # Background subtraction
        fg_mask, gray_frame2 = self.get_background_subtraction_mask(frame2)

        #edge detection
        edge_mask = self.get_edge_mask(gray_frame2, low_thresh=50, high_thresh=150)

        # Combined mask
        final_mask = self.get_final_mask(optical_flow_mask, fg_mask, edge_mask, kernel)

        # Detect contours and bounding boxes
        detections = get_contour_detections(final_mask, thresh=bbox_thresh)
        if len(detections) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        bboxes = detections[:, :4]
        return merge_bounding_boxes(bboxes, treshold=combine_bboxes_thresh)
    

    def single_step_bounding_boxes(self, frame1_path, frame2, kernel, resize_height, resize_width):
        #frame1_path=frame1_path
        #frame2_path=frame2_path
        frame1 = cv2.imread(frame1_path)

        if frame1 is None or frame2 is None:
            return np.zeros((0, 5), dtype=np.float32)
        
        # Resize frames
        frame1 = cv2.resize(frame1, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        return self.get_detections(frame1, frame2, motion_thresh=1, bbox_thresh=400, kernel=kernel)
    

    def main_with_optical_flow(self, frames_dir, output_video, resize_height, resize_width):
        """Main loop to process frames and generate video with detections."""
        out = initialize_video_writer(output_video, fps=15, frame_size=(resize_width, resize_height))
        kernel = np.ones((5, 5), dtype=np.uint8)

        prev_bounding_boxes = [[0, 0, 0, 0]]

        for i in tqdm(range(1, len(gl.glob1(frames_dir, "*.jpg")))):
            frame2_path=f"{frames_dir}/frame{i}.jpg"
            frame2 = cv2.imread(frame2_path)
            frame2 = cv2.resize(frame2, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

            """
            frame1_path=f"{frames_dir}/frame{i-1}.jpg"
            frame2_path=f"{frames_dir}/frame{i}.jpg"
            frame1 = cv2.imread(frame1_path)
            frame2 = cv2.imread(frame2_path)

            if frame1 is None or frame2 is None:
                continue

            # Resize frames
            frame1 = cv2.resize(frame1, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            frame2 = cv2.resize(frame2, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

            # Get detections
            if i % 4 == 0 or i == 1:
                detections = self.get_detections(frame1, frame2, motion_thresh=1, bbox_thresh=400, kernel=kernel)
            """
            if i % 4 == 0 or i == 1:
                detections=self.single_step_bounding_boxes(f"{frames_dir}/frame{i-1}.jpg", frame2, kernel, resize_height, resize_width)
            # Draw bounding boxes
            if detections.size != 0:
                prev_bounding_boxes.append(detections)
                draw_bboxes(frame2, detections)
                prev_bounding_boxes.pop(0)

            # Write to output video
            out.write(frame2)
            cv2.imwrite(f"test/frame{i}.jpg", frame2)

        out.release()

    
combine_tracking=OpticalFlowTracking()
combine_tracking.get_bounded_boxes()