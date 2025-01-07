"""
    General Functions for Motion Detection from a derived Motion Mask

"""

import os
from glob import glob
import numpy as np
import cv2
from PIL import Image



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


def merge_bounding_boxes(boxes, scores, overlapThresh):
    """
    Unisce bounding box che si sovrappongono in modo significativo.
    boxes: lista di tuple (x, y, w, h)
    overlapThresh: soglia di overlap (0.5 = 50%)
    """

    #if len(boxes) == 0:
    #    return []
    
    # Converto in (x1, y1, x2, y2)
    # Sort the boxes by score in descending order
    boxes = boxes[np.argsort(scores)[::-1]]
    
    rects = []
    for (x, y, w, h) in boxes:
        rects.append([x, y, x+w, y+h])
    
    rects = np.array(rects)
    # Ordino in base alla coordinata x
    pick = []
    x1 = rects[:,0]
    y1 = rects[:,1]
    x2 = rects[:,2]
    y2 = rects[:,3]
    idxs = np.argsort(y2)  # ordino per la y massima (o x2, a piacere)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            
            # Calcolo overlap
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            overlapArea = float(w * h)
            
            area1 = float((x2[i] - x1[i]) * (y2[i] - y1[i]))
            area2 = float((x2[j] - x1[j]) * (y2[j] - y1[j]))
            
            #ratio = overlapArea / min(area1, area2)
            ratio=area1+area2-overlapArea
            #print(ratio)
            # Se l'overlap supera la soglia, unisco i due box
            if ratio > 1:
                suppress.append(pos)
                
                # Unione effettiva: aggiorno la box “i” con i min e i max
                x1[i] = min(x1[i], x1[j])
                y1[i] = min(y1[i], y1[j])
                x2[i] = max(x2[i], x2[j])
                y2[i] = max(y2[i], y2[j])
        
        idxs = np.delete(idxs, suppress)
    
    # pick ora contiene gli indici dei box finali
    merged = []
    for i in pick:
        merged.append((x1[i], y1[i], x2[i]-x1[i], y2[i]-y1[i]))
    
    return merged




# =============================================================================
# plot/display utils

def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

'''
def get_color(number):
    """ Converts an integer number to a color """
    # change these however you want to
    blue = int(number*30 % 256)
    green = int(number*103 % 256)
    red = int(number*50 % 256)

    return red, blue, green


def plot_points(image, points, radius=3, color=(0,255,0)):
    for x,y in points:
        cv2.circle(image, (int(x), int(y)), radius, color, thickness=-1)

    return image


def create_gif_from_images(save_path : str, image_path : str, ext : str, duration : int = 50) -> None:
    """ creates a GIF from a folder of images
        Inputs:
            save_path - path to save GIF
            image_path - path where images are located
            ext - extension of the images
        Outputs:
            None
    """
    ext = ext.replace('.', '')
    image_paths = sorted(glob(os.path.join(image_path, f'*.{ext}')))
    image_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    pil_images = [Image.open(im_path) for im_path in image_paths]

    pil_images[0].save(save_path, format='GIF', append_images=pil_images,
                       save_all=True, duration=duration, loop=0)

'''
    