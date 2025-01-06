import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import glob
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

def compute_flow(frame1_path, frame2_path):
    # convert to grayscale
    gray1=load_grayscale_image(frame1_path)
    gray2=load_grayscale_image(frame2_path)

    #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (1366, 768) , interpolation= cv2.INTER_LINEAR)
    gray2 = cv2.resize(gray2, (1366, 768) , interpolation= cv2.INTER_LINEAR)

    # blurr image
    gray1 = cv2.GaussianBlur(gray1, dst=None, ksize=(3,3), sigmaX=5)
    gray2 = cv2.GaussianBlur(gray2, dst=None, ksize=(3,3), sigmaX=5)

    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,
                                        pyr_scale=0.75,
                                        levels=3,
                                        winsize=5,
                                        iterations=3,
                                        poly_n=10,
                                        poly_sigma=1.2,
                                        flags=0)
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


def get_motion_mask(flow_mag, motion_thresh=1, kernel=np.ones((7,7))):
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





def get_detections(frame1, frame2, motion_thresh=1, bbox_thresh=400, nms_thresh=0.1, mask_kernel=np.ones((7,7), dtype=np.uint8)):

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
    #TODO: modificare questo metodo in modo che si possano prendere anche i contours
    detections = get_contour_detections(motion_mask, thresh=bbox_thresh)
    #SI POTREBBE PENSARE DI USARE IL get_contour_detections_adv

    if len(detections) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    
    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]

    # perform Non-Maximal Supression on initial detections
    # TODO: verificare come vengono disegnati i rettangoli senza questo
    return non_max_suppression(bboxes, scores, threshold=nms_thresh)


def main_with_optical_flow(frames_dir, output_video, resize_height, reseize_width):
    #ref_frame_bg = load_grayscale_image(f"{frames_dir}/frame0.jpg")
    #ref_frame_bg = cv2.resize(ref_frame_bg, (1366, 768) , interpolation= cv2.INTER_LINEAR)
    prev_bounding_boxes = [[0,0,0,0]]

   
    out = initialize_video_writer(output_video, fps=15, frame_size=(reseize_width, resize_height))

    #out = initialize_video_writer(output_video, fps=15, frame_size=(3420,1910))

    # get variable motion thresh based on prior knowledge of camera position
    #height, width = mag.shape[:2] 
    #prima erano le dimensioni originali
    motion_thresh = np.c_[np.linspace(0.3, 1, resize_height)].repeat(reseize_width, axis=-1)
    #prima era 7 7
    kernel = np.ones((5,5), dtype=np.uint8)
    #                                                   prima era len -1
    #for i in tqdm(range(151, 250)):
    for i in tqdm(range(1, len(glob.glob1(frames_dir, "*.jpg")))):
        #frame = load_grayscale_image(f"{frames_dir}/frame{i}.jpg")
        #frame = cv2.resize(frame, (1366, 768) , interpolation= cv2.INTER_LINEAR)
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
            # get detections
            detections = get_detections(frame1_bgr_path, 
                                frame2_bgr_path, 
                                motion_thresh=motion_thresh, 
                                bbox_thresh=400, 
                                nms_thresh=0.1, 
                                mask_kernel=kernel)

        # draw bounding boxes on frame
        if detections.size!=0:
            #detections=check_detection(detections)
            prev_bounding_boxes.append(detections)
            draw_bboxes(frame2_bgr, detections)
            prev_bounding_boxes.pop(0)

        out.write(frame2_bgr)
        cv2.imwrite(f"test/frame{i}.jpg", frame2_bgr)  

    out.release()


output_video = "human-detection-optical-flow.avi"
frames_dir="frames"
hight=768
width=1366
main_with_optical_flow(frames_dir, output_video, hight, width)












#get_contour_detections_adv É USUATA. L'INCREMENTO DI PERFORMANCE PRATICAMENTE É INESISTENTE E CI IMPEGA 5 VOLTE TANTO
#STEP 1 E 2 SERVE PER PRENDERE I PARAMETRI NECESSARI

def step1_and_step2(frame_1_path,frame_2_path):
    # compute dense optical flow
    flow = compute_flow(frame_1_path,frame_2_path)

    # separate into magntiude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # get optical flow visualization
    rgb = get_flow_viz(flow)

    '''
    # get variable motion thresh based on prior knowledge of camera position
    motion_thresh = np.c_[np.linspace(0.3, 1, 1080)].repeat(1920, axis=-1)
    '''

    # get variable motion thresh based on prior knowledge of camera position
    height, width = mag.shape[:2] 
    motion_thresh = np.c_[np.linspace(0.3, 1, height)].repeat(width, axis=-1)

    # get motion mask
    mask = get_motion_mask(mag, motion_thresh=motion_thresh)
    return mask, ang



mask, ang= step1_and_step2("path1", "path_2")

'''
Idea alla base: each moving object will have corresponding pixels that move in the same 
direction. So if we have a detection with flow angles that move in multiple directions 
(high variation), then we can remove it because it might not correspond to a real moving 
object. We approach this practically by setting a threshold based on the flow angle standard
deviation of all pixels. If the flow angle standard deviation of a given contour exceeds 
this threshold, then we will not consider it as a detection.
'''

def get_contour_detections_adv(mask, ang=ang, angle_thresh=2, thresh=400):
    """ Obtains initial proposed detections from contours discoverd on the
        mask. Scores are taken as the bbox area, larger is higher.
        Inputs:
            mask - thresholded image mask
            angle_thresh - threshold for flow angle standard deviation
            thresh - threshold for contour size
        Outputs:
            detectons - array of proposed detection bounding boxes and scores 
                        [[x1,y1,x2,y2,s]]
        """
    # get mask contours
    contours, _ = cv2.findContours(mask, 
                                    cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_TC89_L1)
    temp_mask = np.zeros_like(mask) # used to get flow angle of contours
    angle_thresh = angle_thresh*ang.std()
    detections = []
    for cnt in contours:
        # get area of contour
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h

        # get flow angle inside of contour
        cv2.drawContours(temp_mask, [cnt], 0, (255,), -1)
        flow_angle = ang[np.nonzero(temp_mask)]

        if (area > thresh) and (flow_angle.std() < angle_thresh):
            detections.append([x,y,x+w,y+h, area])

    return np.array(detections)
