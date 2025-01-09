"""
    General Functions for Motion Detection from a derived Motion Mask

"""

import os
from glob import glob
import numpy as np
import cv2
from PIL import Image


# ============================================================================
# initialize video

def initialize_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


# =============================================================================
# get bounding box detections from blobs/contours

def get_contour_detections(mask, thresh=400):
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
    detections = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area > thresh: # hyperparameter
            detections.append([x,y,x+w,y+h, area])

    return np.array(detections)



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


# =============================================================================
# Non-Max Supression for detected bounding boxes on blobs

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


def boxes_overlap_area(boxA, boxB):
    """
    Ritorna l'area di intersezione tra due box in formato (x1, y1, x2, y2).
    Se i box non si sovrappongono, l'area sarà 0.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    return inter_width * inter_height

def unify_boxes(boxA, boxB):
    """
    Restituisce il rettangolo "union" che copre entrambi i box
    in formato (x1, y1, x2, y2).
    """
    x1 = min(boxA[0], boxB[0])
    y1 = min(boxA[1], boxB[1])
    x2 = max(boxA[2], boxB[2])
    y2 = max(boxA[3], boxB[3])
    return (x1, y1, x2, y2)



def merge_bounding_boxes(boxes):
        #print("merge function")
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
        
        order = remove_contained_bboxes(boxes)
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order:
                intersection = boxes_overlap_area(boxes[i], boxes[j])
                if intersection > 0:
                    # Unisci i due box in uno
                    unified = unify_boxes(boxes[i], boxes[j])
                    boxes[i] = unified
                    order.remove(j)
                    
        return boxes[keep]


def merge_bounding_boxes_while_loop(bboxes):
    old_len = -1
    # Continuiamo finché il numero di box cambia
    while len(bboxes) != old_len:
        old_len = len(bboxes)
        bboxes = merge_bounding_boxes(bboxes)
    return bboxes
        

















# ============================================================================
# NON PIU' USATE
                    
def non_max_suppression(boxes, scores, threshold=1e-2):
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
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]

    # remove all contained bounding boxes and get ordered index
    order = remove_contained_bboxes(boxes)

    keep = []
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
            #print(iou)
            if iou > threshold:
                order.remove(j)
                
    return boxes[keep]


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
            #print(iou)
            if iou > threshold:
                order.remove(j)

    for i in keep:
        cv2.contourArea(contours[i])

    return boxes[keep], [contours[i] for i in keep]




# =============================================================================
# plot/display utils

def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)