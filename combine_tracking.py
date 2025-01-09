import cv2
#import os
from tqdm import tqdm
#import matplotlib.pyplot as plt
import glob as gl
import numpy as np

from motion_detection_utils import *


def load_grayscale_image(file_path):
    try:
        frame = cv2.imread(file_path)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def compute_flow(frame1_path, frame2_path,                          #in origine
                 pyr_scale=0.5,    # recommended range: [0.3, 0.6]  0.75
                 levels=4,        # recommended range: [3, 6]       3
                 winsize=15,      # recommended range: [5, 21]      5
                 iterations=3,    # recommended range: [3, 10]      3
                 poly_n=7,        # recommended range: [5, 7]       10
                 poly_sigma=1.5,  # recommended range: [1.1, 1.5]   1.2
                 flow_flags=0):

    # convert to grayscale
    gray1=load_grayscale_image(frame1_path)
    gray2=load_grayscale_image(frame2_path)

    #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (768, 576) , interpolation= cv2.INTER_LINEAR)
    gray2 = cv2.resize(gray2, (768, 576) , interpolation= cv2.INTER_LINEAR)

    # blurr image
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

# USATO SOLO PER LA VISUALIZZAZIONE DURANTE LA FASE DI TEST
def get_flow_viz(flow):
    """ Obtains BGR image to Visualize the Optical Flow 
        """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def get_motion_mask(flow_mag, motion_thresh=1, kernel=np.ones((9,), np.uint8)):
    """ Obtains Detection Mask from Optical Flow Magnitude
        Inputs:
            flow_mag (array) Optical Flow magnitude
            motion_thresh - thresold to determine motion
            kernel - kernal for Morphological Operations
        Outputs:
            motion_mask - Binray Motion Mask
        """
    motion_mask = np.uint8(flow_mag > motion_thresh)*255

    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
    #motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    #viene susato la MORPH_CLOSE con lo scopo di rimuovere piccoli spazi o buchi all'interno di oggetti e collegare componenti vicine.
    #così se parti di una persona si muovono più di altre, questo cerca di unirle tutte
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    return motion_mask



##############################
# 3. COMBINED DETECTION     #
##############################
def get_detections(frame1, frame2,
                   motion_thresh=1,
                   bbox_thresh=400,
                   combine_bboxes_thresh=0.1,
                   mask_kernel=np.ones((7,7), dtype=np.uint8)):
    """
    Main function to get detections via combined approach:
      1) Optical Flow
      2) Background Subtraction
      3) Edge Detection
    """

    # --- Step A: Optical Flow & Motion Mask ---
    flow = compute_flow(frame1, frame2)
    #if flow is None:
    #    return np.zeros((0, 5), dtype=np.float32)
    #restituisce l'intensità del movimento
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    #filtra solo i movimenti significativi
    motion_mask = get_motion_mask(mag, motion_thresh=motion_thresh, kernel=mask_kernel)

    # --- Step B: Background Subtraction ---

    bgr_frame2 = cv2.imread(frame2)
    if bgr_frame2 is None:
        return np.zeros((0, 5), dtype=np.float32)

    # Convert to grayscale for edges & also apply BG subtractor
    gray_frame2 = cv2.cvtColor(bgr_frame2, cv2.COLOR_BGR2GRAY)
    #individua background e foreground
    fg_mask = bg_subtractor.apply(bgr_frame2)  # foreground mask from BG subtractor

    # --- Step C: Edge Detection ---
    #individua i bordi nel frame
    edge_mask = get_edge_mask(gray_frame2, low_thresh=50, high_thresh=150)

    # We combine the three masks (motion, background-subtracted, edge) to yield
    # a more robust region of interest:
    #   1) motion_mask  => captures movement from optical flow
    #   2) fg_mask      => highlights new/moving objects vs. the background
    #   3) edge_mask    => focuses on structural boundaries
    combined_mask = cv2.bitwise_and(motion_mask, fg_mask)
    combined_mask = cv2.bitwise_and(combined_mask, edge_mask)

    # Morphological cleanup on the final mask
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, mask_kernel, iterations=1)

    # Convert to binary for contour detection
    final_mask = np.uint8(combined_mask > 0) * 255

    # --- Step D: Get bounding boxes from final mask ---
    detections = get_contour_detections(final_mask, thresh=bbox_thresh)
    if len(detections) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    bboxes = detections[:, :4]
    #scores = detections[:, -1]
    
    # --- Step E: Apply Non-Maximal Suppression ---
    #return non_max_suppression(bboxes, scores, threshold=nms_thresh)
    #return merge_bounding_boxes(bboxes, scores)
    return merge_bounding_boxes(bboxes)

###############################
# 2. EDGE-BASED REFINEMENT   #
###############################
# Adding an optional Canny edge detection step can help refine the regions
# where motion is detected. This can reduce false positives in homogeneous areas
# by focusing on structural features. We’ll combine the edge mask with the motion mask.

def get_edge_mask(gray_frame, low_thresh=100, high_thresh=200):
    """Perform Canny edge detection and return a binary edge mask."""
    edges = cv2.Canny(gray_frame, threshold1=low_thresh, threshold2=high_thresh)
    return edges


#TODO: dato che individua solo alcune componenti di una persona (solo quelle in movimento)
#       questo metodo non funziona bene ← vedere come fixarlo
def check_detection(detections, contours):
    bounding_boxes = []
    for det, cont in zip(detections, contours):
        x,y,w,h = det
        #print(cont)
        area = cv2.contourArea(cont)
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(cont, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        # Filter boxes based on conditions
        if 0.2 < aspect_ratio < 1 and circularity > 0.05 and area > 1500:
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def main_with_optical_flow(frames_dir, output_video, resize_height, reseize_width):
    out = initialize_video_writer(output_video, fps=15, frame_size=(reseize_width, resize_height))

    prev_bounding_boxes = [[0,0,0,0]]
    motion_thresh = np.c_[np.linspace(0.3, 1, resize_height)].repeat(reseize_width, axis=-1)

    #in origine era 7 7
    kernel = np.ones((5,5), dtype=np.uint8)

    for i in tqdm(range(1, len(gl.glob1(frames_dir, "*.jpg")))):
        
        frame1_bgr_path=f"{frames_dir}/frame{i-1}.jpg"
        frame2_bgr_path=f"{frames_dir}/frame{i}.jpg"

        frame1_bgr = cv2.imread(f"{frames_dir}/frame{i-1}.jpg")
        frame2_bgr = cv2.imread(f"{frames_dir}/frame{i}.jpg")

        #prima non cera il resize
        frame1_bgr = cv2.resize(frame1_bgr, (reseize_width, resize_height) , interpolation= cv2.INTER_LINEAR)
        frame2_bgr = cv2.resize(frame1_bgr, (reseize_width, resize_height) , interpolation= cv2.INTER_LINEAR)

        if frame1_bgr is None or frame2_bgr is None:
            continue

        if i % 4 == 0 or i == 1:
            # get detections
            detections = get_detections(frame1_bgr_path, 
                                frame2_bgr_path, 
                                motion_thresh=motion_thresh, 
                                bbox_thresh=400, 
                                combine_bboxes_thresh=0.1, 
                                mask_kernel=kernel)

        # draw bounding boxes on frame
        numpy_det=np.array(detections)
        if numpy_det.size!=0:
            prev_bounding_boxes.append(detections)
            draw_bboxes(frame2_bgr, detections)
            prev_bounding_boxes.pop(0)

        out.write(frame2_bgr)
        cv2.imwrite(f"test/frame{i}.jpg", frame2_bgr)  

    out.release()

if __name__ == "__main__":
    output_video = "human-detection-optical-flow-combined-no-class.avi"
    frames_dir="frames"
    hight=576
    width=768
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200,            # number of frames for background accumulation
        varThreshold=50,        # threshold on squared Mahalanobis distance
        detectShadows=True      # keep it True or False depending on whether you want shadows
    )
    main_with_optical_flow(frames_dir, output_video, hight, width)


    

