import numpy as np
import cv2


def get_contour_detections(mask):
    """ 
        Obtains detections from contours discoverd on the  mask. 
        
        Parameters:
            mask: Binary mask of a frame

        Returns:
            detections: Array of bounding boxes
    """
    # get mask contours
    contours, _ = cv2.findContours(mask, 
                                   cv2.RETR_EXTERNAL,  
                                   cv2.CHAIN_APPROX_TC89_L1)
    detections = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        detections.append([x,y,w+x,h+y])
            
    return np.array(detections)


def boxes_overlap_area(boxA, boxB):
    """
    Compute the intersecation area of two bounding boxes

    Parameters:
        boxA: First bounding box
        boxB: Second bounding box

    Returns:
        IoU: Intersection over Union of the two boxes
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)

    inter_area = inter_width * inter_height

    # Compute the area of the two bounding boxes
    area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_boxB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Comput union area
    union_area = area_boxA + area_boxB - inter_area

    #compute Intersecation over Union
    return inter_area / union_area

def unify_boxes(boxA, boxB):
    """
    Merge two bounding boxes into one

    Parameters:
        boxA: First bounding box
        boxB: Second bounding box

    Returns:
        Unified bounding box
    """
    x1 = min(boxA[0], boxB[0])
    y1 = min(boxA[1], boxB[1])
    x2 = max(boxA[2], boxB[2])
    y2 = max(boxA[3], boxB[3])
    return (x1, y1, x2, y2)



def remove_contained_bboxes(boxes):
    """
    Remove smaller bounding boxes that are completely contained within larger boxes

    Parameters:
        boxes: List of bounding boxes

    Returns:
        List of indices of the remaining bounding boxes
    """
    check_array = np.array([True, True, False, False])
    keep = list(range(0, len(boxes)))
    for i in keep:
        for j in range(0, len(boxes)):
            # check if box j is completely contained in box i
            if np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep


def merge_bounding_boxes(boxes,treshold ,need_check_validity=False):
    """
    Merge overlapping bounding boxes above a certain IoU threshold.
    If need_check_validity = True it also checks if the merged bounding 
    boxes still contains a person

    Parameters:
        boxes: List of bounding boxes
        threshold: IoU threshold for merging boxes
        need_check_validity: If check the unified bounding box need to be checked

    Returns:
        Updated list of bounding boxes
    """
    order = remove_contained_bboxes(boxes)
    boxes=np.array(boxes)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        for j in order:
            intersection = boxes_overlap_area(boxes[i], boxes[j])
            if intersection > treshold:
                # If intersecation over treshold, unify the boxes
                unified = unify_boxes(boxes[i], boxes[j])
                # Used to avoid false positive
                if need_check_validity:
                    if is_valid_box(unified[2], unified[3]):
                        boxes[i] = unified
                        order.remove(j)
                else:
                    boxes[i] = unified
                    order.remove(j)
    return boxes[keep]


def merge_bounding_boxes_while_loop(bboxes, treshold, check_validity=False):
    """
    Iteratively merge bounding boxes until no further merges are possible.

    Parameters:
        bboxes: List of bounding boxes
        threshold: IoU threshold for merging boxes
        check_validity: Whether to validate the unified box

    Returns:
        Final list of merged bounding boxes
    """
    old_len = -1
    # Continue to merge until no more boxes are merged (len of previous and current boxes are equal)
    while len(bboxes) != old_len:
        old_len = len(bboxes)
        bboxes = merge_bounding_boxes(bboxes, treshold, check_validity)
    return bboxes


def preprocess_frames(frame):
    """
    Preprocess a frame by converting it to grayscale and resizing

    Parameters:
        frame: Input frame

    Returns:
        Processed frame
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (1366, 768), interpolation=cv2.INTER_LINEAR)
    return frame

def is_valid_box(width, height):
    """
    Validate a bounding box contains a person

    Parameters:
        width: Width of the bounding box
        height: Height of the bounding box

    Returns:
        True if the bounding box is valid, False otherwise.
    """
    area = width * height
    aspect_ratio = float(width) / height
    return 0.2 < aspect_ratio < 1 and area > 1500