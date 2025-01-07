# Import Necessary Libraries

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import time
import tqdm
import glob as gl


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


def main_real_tracking(frames_dir, output_video, resize_height, reseize_width):
    out = initialize_video_writer(output_video, fps=15, frame_size=(reseize_width, resize_height))

    prev_bounding_boxes = [[0,0,0,0]]

    for i in tqdm(range(1, len(gl.glob1(frames_dir, "*.jpg")))):
        
        frame1_bgr_path=f"{frames_dir}/frame{i-1}.jpg"
        #frame2_bgr_path=f"{frames_dir}/frame{i}.jpg"

        frame1_bgr = cv2.imread(f"{frames_dir}/frame{i-1}.jpg")
        #frame2_bgr = cv2.imread(f"{frames_dir}/frame{i}.jpg")

        #prima non cera il resize
        frame1_bgr = cv2.resize(frame1_bgr, (reseize_width, resize_height) , interpolation= cv2.INTER_LINEAR)
        #frame2_bgr = cv2.resize(frame1_bgr, (reseize_width, resize_height) , interpolation= cv2.INTER_LINEAR)

        if frame1_bgr is None:
            continue


if __name__ == "__main__":
    output_video = "human-detection-tracking.avi"
    frames_dir="frames"
    hight=480
    width=512
    # I am giving  big random numbers for x_min and y_min because if you initialize them as zeros whatever coordinate you go minimum will be zero 
    x_min,y_min,x_max,y_max=36000,36000,0,0
    main_real_tracking(frames_dir, output_video, hight, width)