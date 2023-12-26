import cv2
from pathlib import Path
from tqdm import tqdm
from os import makedirs
import numpy as np


def make_video(input_path: str, output_path: str, first_frame: int, last_frame: int, original_video=None, slowmotion=1):

    padding = 10
    input_path = Path(input_path)
    if original_video is not None:
        original_video = Path(original_video)

    image = cv2.imread(str(input_path / f'best_frame_{first_frame}_0.png'))
    size = image.shape[:2]
    if original_video is not None:
         size = (size[0] * 2, size[1])
    
    h_mosaic = 2 * image.shape[0] + 3 * padding
    mosaic = np.uint8(255 * np.ones((h_mosaic, padding, image.shape[2])))

    outcrop = cv2.VideoWriter(output_path, -1,
                              30, size)
    
    for frame_num in range(first_frame, last_frame):
        for subframe in range(slowmotion):
            image = cv2.imread(str(input_path / f'best_frame_{frame_num}_{subframe}.png'))
            if original_video is not None:
                if subframe == 0 and (original_video / f'frame{frame_num}.png').exists():
                    orig_image = cv2.imread(str(original_video / f'frame{frame_num}.png'))
                    if (frame_num - first_frame) % 5 == 0 and frame_num - first_frame < 47:
                        mosaic = cv2.hconcat([mosaic,
                                            np.uint8(255 * np.ones((h_mosaic, padding, mosaic.shape[2]))),
                                            cv2.vconcat([np.uint8(255 * np.ones((padding, image.shape[1], mosaic.shape[2]))),
                                                        cv2.resize(orig_image, image.shape[:2]),
                                                        np.uint8(255 * np.ones((padding, image.shape[1], mosaic.shape[2]))),
                                                        image,
                                                        np.uint8(255 * np.ones((padding, image.shape[1], mosaic.shape[2])))])
                                            ])           
                image = cv2.hconcat([image, cv2.resize(orig_image, image.shape[:2])])
            print(frame_num)
            outcrop.write(image)
    mosaic = cv2.hconcat([mosaic, np.uint8(255 * np.ones((h_mosaic, padding, mosaic.shape[2])))])
    cv2.imwrite(str(input_path / '..' / 'mosaic.png'), mosaic)
    outcrop.release()

if __name__ == "__main__":
    dir_path = r"C:\projects\VideoSketch\results\kangaroo\kangaroo_5_curves_5k_centerloss_centeriter_motionloss_onecycle_level_5_centerlevel_7_model_ver_1_pos_enc_9_anchormorecenter_28\images"
    output_path = r'C:\projects\VideoSketch\results\kangaroo\kangaroo_5_curves_5k_centerloss_centeriter_motionloss_onecycle_level_5_centerlevel_7_model_ver_1_pos_enc_9_anchormorecenter_28\best_iter_video.mp4'
    original_video = r'C:\projects\VideoSketch\preped_videos\kangaroo'
    slowmotion = 6
    make_video(dir_path, output_path, 1, 146,
               original_video, slowmotion)