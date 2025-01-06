import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import glob
import numpy as np


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


def remove_contained_bboxes(boxes):
    """ Removes all smaller boxes that are contained within larger boxes.
        Requires bboxes to be soirted by area (score)
        Inputs:
            boxes - array bounding boxes sorted (descending) by area 
                    [[x1,y1,x2,y2]]
        Outputs:
            keep - indexes of bounding boxes that are not entirely contained 
                   in another box
    """
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep: # range(0, len(bboxes)):
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep


def non_max_suppression_edit(boxes, scores, contours, threshold=1e-1):
    """
    Perform non-max suppression on a set of bounding boxes 
    and corresponding scores.
    Inputs:
        boxes: a list of bounding boxes in the format [xmin, ymin, xmax, ymax]
        scores: a list of corresponding scores 
        threshold: the IoU (intersection-over-union) threshold for merging bboxes
    Outputs:
        boxes - non-max suppressed boxes
    """
    for cont in contours:
        cv2.contourArea(cont)

    # Sort the boxes and contours by score in descending order
    order_ind = np.argsort(scores)[::-1]
    boxes = boxes[order_ind]
    #contours=contours[order]
    contours = [contours[i] for i in order_ind]

    # Remove all contained bounding boxes and get ordered indices
    order = remove_contained_bboxes(boxes)

    # Filter boxes and contours based on `keep_indices`
    #boxes = boxes[keep_indices]
    #contours[keep_indices]
    #contours = [contours[i] for i in keep_indices]

    keep = []
    #final_contours = []

    while order:

        i = order.pop(0)
        keep.append(i)
        for j in order:
            # Calculate the IoU between the two boxes
            intersection = max(0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) * \
                           max(0, min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1]))
            union = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) + \
                    (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1]) - intersection
            iou = intersection / union

            # Remove boxes with IoU greater than the threshold
            if iou > threshold:
                order.remove(j)

    for i in keep:
        cv2.contourArea(contours[i])

    return boxes[keep], [contours[i] for i in keep]


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
    #TODO: modificare questo metodo in modo che si possano prendere anche i contours
    detections, contours = get_contour_detections_edit(motion_mask, thresh=bbox_thresh)

    if len(detections) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    
    # separate bboxes and scores
    bboxes = detections[:, :4]
    scores = detections[:, -1]

    # perform Non-Maximal Supression on initial detections
    # TODO: verificare come vengono disegnati i rettangoli senza questo
    return non_max_suppression_edit(bboxes, scores, contours, threshold=nms_thresh)
    #return non_max_suppression(bboxes, scores, threshold=nms_thresh)


def get_contour_detections_edit(mask, thresh=400):
    """ Obtains initial proposed detections from contours discoverd on the
        mask. Scores are taken as the bbox area, larger is higher.
        Inputs:
            mask - thresholded image mask
            thresh - threshold for contour size
        Outputs:
            detectons - array of proposed detection bounding boxes and scores 
                        [[x1,y1,x2,y2,s]]
        """
    # get mask contours
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL, # cv2.RETR_TREE, 
                                   cv2.CHAIN_APPROX_TC89_L1)
    for cont in contours:
        area=cv2.contourArea(cont)

    detections = []
    filtered_contours = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area > thresh: # hyperparameter
            detections.append([x,y,x+w,y+h, area])
            filtered_contours.append(cnt)

    return np.array(detections), filtered_contours

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


def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)


def main_with_optical_flow_edit(frames_dir, output_video, resize_height, reseize_width):
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
    #for i in tqdm(range(50, 749)):
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
            '''
            # get detections
            detections, contours = get_detections_edit(frame1_bgr_path, 
                                frame2_bgr_path, 
                                motion_thresh=motion_thresh, 
                                bbox_thresh=400, 
                                nms_thresh=0.1, 
                                mask_kernel=kernel)
            '''
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


output_video = "human-detection-optical-flow-edit.avi"
frames_dir="frames"
hight=768
width=1366
main_with_optical_flow_edit(frames_dir, output_video, hight, width)
