import cv2
from pathlib import Path
from tqdm import tqdm
import os
from os import makedirs
import numpy as np


class FramesDir:
    def __init__(self, dir_path, image_type='jpg', zero_pad=5):
        self.dir_path = Path(dir_path)
        self.image_type = image_type
        self.frame = 0
        self.zero_pad = zero_pad

    def set(self, mode, frame):
        if mode == cv2.CAP_PROP_POS_FRAMES:
            self.frame = frame
    
    def get(self, mode):
        if mode == cv2.CAP_PROP_FRAME_COUNT:
            return len(list(self.dir_path.glob(f'*.{self.image_type}')))
        elif mode == cv2.CAP_PROP_FPS:
            return 30
    
    def read(self):
        image = cv2.imread(str(self.dir_path / f'{str(self.frame).zfill(self.zero_pad)}.{self.image_type}'))
        self.frame += 1
        return 0, image
    
    def release(self):
        return


def prep_video(input_path: str, output_dir: str, name: str, start_frame: int, end_frame: int,
               resize_to: int, crop=None):
    if os.path.isdir(input_path):
        cap = FramesDir(input_path)
    else:
        cap = cv2.VideoCapture(input_path)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert start_frame >= 0 and start_frame < end_frame and end_frame < frame_count, \
    f'{start_frame} >= 0 and {start_frame} < {end_frame} and {end_frame} < {frame_count}'
     
    
    output_dir = Path(output_dir)
    if not output_dir.exists():
        makedirs(str(output_dir))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success, image = cap.read()
    
    if crop is not None:
        h_start, h_end, w_start, w_end = crop
        assert h_end - h_start == w_end - w_start, f"{h_end - h_start} == {w_end - w_start}"
        temp = image[h_start:h_end, w_start:w_end]
    im_size = [h_end - h_start, w_end - w_start]
    resize_scale = resize_to / max(im_size)
    size = tuple([int(dim * resize_scale) for dim in im_size[::-1]])

    outcrop = cv2.VideoWriter(str(Path(output_dir) / f"{name}.mp4"), -1,
                              cap.get(cv2.CAP_PROP_FPS), size)
    for count, _ in enumerate(range(start_frame, end_frame)):
        if crop is not None:
            h_start, h_end, w_start, w_end = crop
            image = image[h_start:h_end, w_start:w_end]
            im_h, im_w = image.shape[:2]
            if im_h < h_end - h_start:
                delta = h_end - h_start - im_h
                temp = np.zeros((h_end - h_start, w_end - w_start, image.shape[2]), dtype=np.uint8)
                temp[delta//2:im_h + delta//2, :, :] = image
                image = temp
            if im_w < w_end - w_start:
                delta = w_end - w_start - im_w
                temp = np.zeros((h_end - h_start, w_end - w_start, image.shape[2]))
                temp[:, delta//2:im_w + delta//2, :] = image
                image = temp
        im_size = image.shape[:2]
        resize_scale = resize_to / max(im_size)
        image = cv2.resize(image, [int(dim * resize_scale) for dim in im_size[::-1]])
        outcrop.write(image)
        cv2.imwrite(str(output_dir / f"frame{count}.png"), image)     # save frame as JPEG file      
        success, image = cap.read()
        print('Read a new frame: ', success)
    
    cap.release()
    outcrop.release()

if __name__ == "__main__":
    vid_path = r'C:\projects\VideoSketch\videos\DAVIS\JPEGImages\Full-Resolution\breakdance'  # r'C:\projects\VideoSketch\videos\pexels_videos_1171808 (720p).mp4'
    out_dir = r'C:\projects\VideoSketch\preped_videos\breakdance'
    name = 'breakdance'
    prep_video(vid_path, out_dir, name, start_frame=0, end_frame=59, resize_to=500,
               crop=(0, 1079, 1920-1080-500, 1919-500))