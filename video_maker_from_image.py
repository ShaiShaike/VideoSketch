import cv2
from pathlib import Path
from tqdm import tqdm
from os import makedirs


def make_video(input_path: str, output_dir: str):

    output_dir = Path(output_dir)
    if not output_dir.exists():
        makedirs(str(output_dir))
    
    image = cv2.imread(input_path)
    size = image.shape[:2]

    outcrop = cv2.VideoWriter(str(Path(output_dir) / "ballerina.mp4"), -1,
                              30, size)
    for count, _ in enumerate(range(0, 30)):
        outcrop.write(image)
    
    outcrop.release()

if __name__ == "__main__":
    image_path = r"C:\projects\VideoSketch\SceneSketch\target_images\scene\ballerina.png"
    out_dir = r'C:\projects\VideoSketch\preped_videos\ballerina_image'
    make_video(image_path, out_dir)