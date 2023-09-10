import cv2
from pathlib import Path
from tqdm import tqdm
from os import makedirs

def prep_video(input_path: str, output_dir: str, start_frame: int, end_frame: int,
               resize_to: int, crop=None):
    cap = cv2.VideoCapture(input_path)
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert start_frame >= 0 and start_frame < end_frame and end_frame < frame_count
    
    output_dir = Path(output_dir)
    if not output_dir.exists():
        makedirs(str(output_dir))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    success,image = cap.read()
    
    if crop is not None:
        h_start, h_end, w_start, w_end = crop
        temp = image[h_start:h_end, w_start:w_end]
    im_size = temp.shape[:2]
    resize_scale = resize_to / max(im_size)
    size = tuple([int(dim * resize_scale) for dim in im_size[::-1]])

    outcrop = cv2.VideoWriter(str(Path(output_dir) / "ballerina2.mp4"), -1,
                              cap.get(cv2.CAP_PROP_FPS), size)
    for count, _ in enumerate(range(start_frame, end_frame)):
        if crop is not None:
            h_start, h_end, w_start, w_end = crop
            image = image[h_start:h_end, w_start:w_end]
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
    vid_path = r'C:\projects\VideoSketch\videos\production_id_4990428 (1080p).mp4'
    out_dir = r'C:\projects\VideoSketch\preped_videos\ballerina2'
    prep_video(vid_path, out_dir, start_frame=7*30, end_frame=9*30, resize_to=500,
               crop=(690, 1330, 170, 810))