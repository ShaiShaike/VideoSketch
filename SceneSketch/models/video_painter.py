import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from os import makedirs
import CLIP_.clip as clip
from torchvision import transforms
import torch
import sketch_utils as utils

from painter_params import Painter


class VideoPainter(Painter):
    def __init__(self, workdir, args, num_strokes=4, num_segments=4, imsize=224, device=None):
        super().__init__(args, num_strokes, num_segments, imsize, device, None, None)

        self.workdir = Path(workdir)
        if "for" in args.loss_mask:
            # default for the mask is to mask out the background
            # if mask loss is for it means we want to maskout the foreground
            self.reverse_mask = True
        else:
            self.reverse_mask = False
            
    
    def load_clip_attentions_and_mask(self, frame_index):
        attentions_path = self.workdir / f"clip_attentions_{frame_index}.t"
        mask_path = self.workdir / f"mask_{frame_index}.npy"
        assert attentions_path.exists() and mask_path.exists()
        self.image_input_attn_clip = torch.load(str(attentions_path)).to(self.device)

        mask = np.load(str(mask_path))
        self.mask = 1- mask if self.reverse_mask else mask
        if self.attention_init:
            torch.load(str(self.workdir / f"attention_map_{frame_index}.t")).to(self.device)
        else:
            self.attention_map = None

    def get_target(self, frame_index):
        target_path = self.workdir / f"frame_{frame_index}.png"
        assert target_path.exists()
        return Image.open(str(target_path))
    
    def prep_video_inputs(self, args, video_path, start_frame, end_frame, resize_to, crop=None):
        # vid_path = r'C:\projects\VideoSketch\videos\production_id_4990428 (1080p).mp4'
        # out_dir = r'C:\projects\VideoSketch\preped_videos\ballerina'

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert start_frame >= 0 and start_frame < end_frame and end_frame < frame_count
        
        if not self.workdir.exists():
            makedirs(str(self.workdir))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        success, image = cap.read()
        for frame_index in range(start_frame, end_frame):
            
            target, mask = self.process_image(image, args, resize_to, crop)
            target.save(str(self.workdir / f"frame_{frame_index}.png"))
            np.save(str(self.workdir / f"mask_{frame_index}.npy"), mask)
            
            clip_attentions = self.clip_it(self, target)
            torch.save(clip_attentions, str(self.workdir / f"clip_attentions_{frame_index}.t"))

            if self.attention_init:
                self.image_input_attn_clip = clip_attentions
                attention_map = self.set_attention_map()
                torch.save(attention_map, str(self.workdir / f"attention_map_{frame_index}.t"))
            else:
                self.attention_map = None
            
            success, image = cap.read()
    
    def process_image(self, image, args, resize_to, crop):
        if crop is not None:
            h_start, h_end, w_start, w_end = crop
            image = image[h_start:h_end, w_start:w_end]
        im_size = image.shape[:2]
        resize_scale = resize_to / max(im_size)
        image = cv2.resize(image, [int(dim * resize_scale) for dim in im_size[::-1]])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = Image.fromarray(image)

        if target.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", target.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")
        
        masked_im, mask = utils.get_mask_u2net(args, target)
        
        if args.mask_object:
            target = masked_im
        if args.fix_scale:
            target = utils.fix_image_scale(target)

        transforms_ = []
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
        transforms_.append(transforms.ToTensor())
        data_transforms = transforms.Compose(transforms_)

        target_ = data_transforms(target).unsqueeze(0).to(args.device)
        mask = Image.fromarray((mask*255).astype(np.uint8)).convert('RGB')
        mask = data_transforms(mask).unsqueeze(0).to(args.device)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        return target_, mask

    def clip_it(self, image):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
                    preprocess.transforms[-1],
                ])
        return data_transforms(image)

