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
    
def initialize_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

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

    gray1 = cv2.resize(gray1, (512, 480) , interpolation= cv2.INTER_LINEAR)
    gray2 = cv2.resize(gray2, (512, 480) , interpolation= cv2.INTER_LINEAR)

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

                                                              #7,7 senza np.uint8 in origine
def get_motion_mask(flow_mag, motion_thresh=1, kernel=np.ones((5,5), np.uint8)):
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
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return motion_mask


def get_detections_edit(frame1, frame2, motion_thresh=1, bbox_thresh=400, nms_thresh=0.1, mask_kernel=np.ones((7,7), dtype=np.uint8)):

    """ Main function to get detections via Frame Differencing
        Inputs:
            frame1 - Grayscale frame at time t
            frame2 - Grayscale frame at time t + 1
            motion_thresh - Minimum flow threshold for motion
            bbox_thresh - Minimum threshold area for declaring a bounding box 
            nms_thresh - IOU threshold for computing Non-Maximal Supression
            mask_kernel - kernel for morphological operations on motion mask
        Outputs:
            detections - list with bounding box locations of all detections
                bounding boxes are in the form of: (xmin, ymin, xmax, ymax)
        """
    
    
    # get optical flow
    flow = compute_flow(frame1, frame2)

    # separate into magntiude and angle
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    motion_mask = get_motion_mask(mag, motion_thresh=motion_thresh, kernel=mask_kernel)

    # get initially proposed detections from contours
    detections, contours = get_contour_detections_edit(motion_mask, thresh=bbox_thresh)

    if len(detections) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    
    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]

    # perform Non-Maximal Supression on initial detections

    # TODO: verificare come vengono disegnati i rettangoli senza questo
    return non_max_suppression_edit(bboxes, scores, contours, threshold=nms_thresh)


def check_detection(detections, contours):
    """
    bounding_boxes = []
    for det, cont in zip(detections, contours):
        x,y,w,h = det
        # Evita divisioni per zero (ad esempio se h=0)
        if h == 0 or w == 0:
            continue
        
        area = cv2.contourArea(cont)            # Area del contorno
        bbox_area = float(w * h)               # Area del bounding box
        aspect_ratio = float(w) / (h + 1e-6)    # Rapporto larghezza / altezza
        perimeter = cv2.arcLength(cont, True)  # Perimetro del contorno

        # Circolarità: 1 indica un cerchio perfetto; valori più bassi forme più allungate
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Extent: area del contorno / area del bounding box
        extent = area / bbox_area if bbox_area > 0 else 0

        # Solidity: area del contorno / area dell'hull convesso
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0


        # Filter boxes based on conditions
        # Filtri euristici per riconoscere persone:
        # 1. Aspect ratio compreso in un range "umanoide"
        # 2. Circolarità non troppo elevata (forme troppo circolari sono sospette)
        # 3. Area minima sufficiente (filtra rumore e oggetti piccoli)
        # 4. Extent in un range: se è troppo vicino a 1 indica contorni "pieni" (potrebbe essere un oggetto compatto)
        # 5. Solidity in un range: se troppo bassa indica contorni molto frastagliati (spesso rumore)
        if (
            0.2 < aspect_ratio < 1.0 and        # persona in piedi: più alta che larga
            0.05 < circularity < 0.5 and        # circolarità "intermedia"
            area > 1500 and                    # area minima
            0.2 < extent < 0.9 and             # contorno non troppo pieno / non troppo "vuoto"
            solidity > 0.5                     # contorno sufficientemente compatto
        ):
            bounding_boxes.append((x, y, w, h))
    """
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


def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)


def main_with_optical_flow_edit(frames_dir, output_video, resize_height, reseize_width):
     
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

        if i % 10 == 0 or i == 1:
            detections, contours = get_detections_edit(frame1_bgr_path, 
                                frame2_bgr_path, 
                                motion_thresh=motion_thresh, 
                                bbox_thresh=400, 
                                nms_thresh=0.1, 
                                mask_kernel=kernel)
        
        # draw bounding boxes on frame
        numpy_det=np.array(detections)
        if numpy_det.size!=0:
            detections=check_detection(detections, contours)
            prev_bounding_boxes.append(detections)
            draw_bboxes(frame2_bgr, detections)
            prev_bounding_boxes.pop(0)

        out.write(frame2_bgr)
        cv2.imwrite(f"test/frame{i}.jpg", frame2_bgr)  

    out.release()

if __name__ == "__main__":
    output_video = "human-detection-optical-flow-edit-v2.avi"
    frames_dir="frames"
    hight=480
    width=512
    main_with_optical_flow_edit(frames_dir, output_video, hight, width)
