import cv2
import numpy as np
import glob
from tqdm import tqdm
import os

def initialize_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def load_grayscale_image(file_path):
    try:
        frame = cv2.imread(file_path)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def process_frame_differences(ref_frame_bg, frame):
    mask = cv2.subtract(ref_frame_bg, frame)
    _, thresh = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    return contours

def get_bounding_boxes(large_contours):
    bounding_boxes = []
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = float(w) / h
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        # Filter boxes based on conditions
        if 0.2 < aspect_ratio < 1 and circularity > 0.05 and area > 1500:
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes


def main(frames_dir, output_video):
    ref_frame_bg = load_grayscale_image(f"{frames_dir}/frame0.jpg")
    ref_frame_bg = cv2.resize(ref_frame_bg, (1366, 768) , interpolation= cv2.INTER_LINEAR)
   
    out = initialize_video_writer(output_video, fps=15, frame_size=(1366, 768))
    prev_bounding_boxes = [[0,0,0,0]]
    for i in tqdm(range(1, len(glob.glob1(frames_dir, "*.jpg")) - 1)):
        frame = load_grayscale_image(f"{frames_dir}/frame{i}.jpg")
        frame = cv2.resize(frame, (1366, 768) , interpolation= cv2.INTER_LINEAR)
        if frame is None:
            continue

        if i % 5 == 0 or i == 1:
            large_contours = process_frame_differences(ref_frame_bg, frame)
            bounding_boxes = get_bounding_boxes(large_contours)

        frame_out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        prev_bounding_boxes.append(bounding_boxes)
        for box in prev_bounding_boxes[-1]:
            x, y, w, h = box
            frame_out = cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 2)
        prev_bounding_boxes.pop(0)
                

        out.write(frame_out)
        cv2.imwrite(f"test/frame{i}.jpg", frame_out)  

    out.release()

if __name__ == "__main__":
    frames_dir = "preprocessed-frames"
    output_video = "human-detection.avi"
    main(frames_dir, output_video)