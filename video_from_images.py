import cv2
from pathlib import Path
from tqdm import tqdm
from os import makedirs


def make_video(input_path: str, output_path: str, first_frame: int, last_frame: int):

    input_path = Path(input_path)

    image = cv2.imread(str(input_path / f'best_iter_frame_{first_frame}.png'))
    size = image.shape[:2]

    outcrop = cv2.VideoWriter(output_path, -1,
                              30, size)
    for frame_num in range(first_frame, last_frame):
        image = cv2.imread(str(input_path / f'best_iter_frame_{frame_num}.png'))
        outcrop.write(image)
    
    outcrop.release()

if __name__ == "__main__":
    dir_path = r"C:\projects\VideoSketch\results\58_frames_motionMLP_on_same_shape_as_MLP\images"
    output_path = r'C:\projects\VideoSketch\results\58_frames_motionMLP_on_same_shape_as_MLP\best_iter_video.mp4'
    make_video(dir_path, output_path, 1, 58)