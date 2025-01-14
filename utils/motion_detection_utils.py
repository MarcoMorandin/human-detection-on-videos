import numpy as np
import cv2

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
            #detections.append([x,y,w,h, area])
            detections.append([x,y,w+x,h+y, area])
            
    #return detections
    return np.array(detections)



# =============================================================================
# Merge bounding boxes



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
    inter_area = inter_width * inter_height

    # Calcola le aree dei due bounding boxes
    area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_boxB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calcola l'area dell'unione
    union_area = area_boxA + area_boxB - inter_area

    return inter_area / union_area

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


def check(x, y, w, h):
    cnt = np.array([
            [[x, y]],  # Vertice in alto a sinistra
            [[w, y]],  # Vertice in alto a destra
            [[w, h]],  # Vertice in basso a destra
            [[x, h]]   # Vertice in basso a sinistra
            ], dtype=np.int32)
    area = cv2.contourArea(cnt)
    aspect_ratio = float(w) / h
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    if 1.5 <= aspect_ratio <= 4 and 2000 <= area and 0.56 <= circularity <= 0.8:
        return True
    return False


def merge_bounding_boxes(boxes, need_check_validity, treshold=1):
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
        boxes=np.array(boxes)
        #return boxes[order]
        
        #order = list(range(0, len(boxes)))
        #Se serve è possibile creare un contour a partire dei 4 valori delle box
        keep = []
        while order:
            i = order.pop(0)
            keep.append(i)
            for j in order:
                intersection = boxes_overlap_area(boxes[i], boxes[j])
                if intersection > treshold:
                    # Unisci i due box in uno
                    unified = unify_boxes(boxes[i], boxes[j])
                    if need_check_validity:
                        if check(unified[0], unified[1], unified[2], unified[3]):
                            boxes[i] = unified
                            order.remove(j)
                    else:
                        boxes[i] = unified
                        order.remove(j)
        return boxes[keep]


def merge_bounding_boxes_while_loop(bboxes, check_validity=False, treshold=1):
    #print(bboxes)
    old_len = -1
    # Continuiamo finché il numero di box cambia
    while len(bboxes) != old_len:
        old_len = len(bboxes)
        bboxes = merge_bounding_boxes(bboxes, check_validity, treshold)
    return bboxes
        


# ============================================================================
# initialize video ← DA ELIMINARE

def initialize_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


# =============================================================================
# plot/display utils

def draw_bboxes(frame, detections):
    for det in detections:
        x1,y1,x2,y2 = det
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,), 2)
