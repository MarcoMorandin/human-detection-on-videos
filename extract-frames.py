
import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_per_second=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video = cv2.VideoCapture(video_path)
    duration = video.get(cv2.CAP_PROP_FRAME_COUNT) // video.get(cv2.CAP_PROP_FPS)

    for i in tqdm(range(int(duration * frame_per_second))):
        video.set(cv2.CAP_PROP_POS_MSEC,((i*1000) / frame_per_second))
        success,image = video.read()
        cv2.imwrite(f"{output_dir}/frame{i}.jpg", image)
        

def preprocess_frames(frames_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame in tqdm(os.listdir(frames_dir)):
        if frame.endswith(".jpg"):
            image = cv2.imread(f"{frames_dir}/{frame}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # used to remove noise
            blur = cv2.GaussianBlur(gray, (0,0), sigmaX=2, sigmaY=2)
            cv2.imwrite(f"{output_dir}/{frame}", blur)    


def preprocess_frames_no_gaussian(frames_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame in tqdm(os.listdir(frames_dir)):
        if frame.endswith(".jpg"):
            image = cv2.imread(f"{frames_dir}/{frame}")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # used to remove noise
            #blur = cv2.GaussianBlur(gray, (0,0), sigmaX=2, sigmaY=2)
            cv2.imwrite(f"{output_dir}/{frame}", gray)  


if __name__ == "__main__":
    print("\nExtracting frames...\n")
    extract_frames("video_2.mov", "frames", 15)
    print("\n\nPreprocessing frames...\n")
    #preprocess_frames("frames", "preprocessed-frames")
    preprocess_frames_no_gaussian("frames", "preprocessed-frames-no-gaussian")

    