import cv2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from time import time
from os import makedirs
import CLIP_.clip as clip
from torchvision import transforms
import torch
import sketch_utils as utils

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.painter_params import Painter


class Timer:
    def __init__(self) -> None:
        self.sum_time = 0
        self.amount = 0
        self.start_timer = None
    
    def tic(self):
        self.start_timer = time()

    def toc(self):
        self.sum_time += time() - self.start_timer()
        self.amount += 1
        self.start_timer = None
    
    def __repr__(self) -> str:
        return f'total time: {self.sum_time :.1f}sec, avarege time: {self.sum_time / self.amount :.1f}sec'



class VideoPainter(Painter):
    def __init__(self, args, num_strokes=4, num_segments=4, imsize=224, device=None):
        super().__init__(args, num_strokes, num_segments, imsize, device, None, None, is_video=args.model_ver)

        self.workdir = Path(args.workdir)
        if "for" in args.loss_mask:
            # default for the mask is to mask out the background
            # if mask loss is for it means we want to maskout the foreground
            self.reverse_mask = True
        else:
            self.reverse_mask = False
        
        self.prep_timer = Timer()
        self.prep_video_inputs(args)
        print('prep time:', self.prep_timer)
        return

        self.base_frame = (args.start_frame + args.end_frame) // 2 if args.center_frame < 0 else args.center_frame
        self.load_clip_attentions_and_mask(self.base_frame)
        self.attention_map = self.set_attention_map() if self.attention_init else None
        self.thresh = self.set_attention_threshold_map() if self.attention_init else None

    def get_motion_mlp(self):
        return self.motion_mlp
    
    def load_clip_attentions_and_mask(self, frame_index):
        self.frame_num = frame_index
        attentions_path = self.workdir / f"clip_attentions_{frame_index}.t"
        mask_path = self.workdir / f"mask_{frame_index}.t"
        assert attentions_path.exists() and mask_path.exists(), str(attentions_path)
        self.image_input_attn_clip = torch.load(str(attentions_path)).to(self.device)

        mask = torch.load(str(mask_path)).to(self.device)
        self.mask = 1- mask if self.reverse_mask else mask
        if self.attention_init:
            self.attention_map = np.load(str(self.workdir / f"attention_map_{self.base_frame}.npy"))
        else:
            self.attention_map = None
    
    def get_edges(self, frame_index):
        edges_path = self.workdir / f"edges_{frame_index}.t"
        assert edges_path.exists(), f"{str(edges_path)} does not exists"
        return torch.load(str(edges_path)).to(self.device)

    def get_target(self, frame_index):
        target_path = self.workdir / f"frame_{frame_index}.t"
        assert target_path.exists()
        return torch.load(str(target_path)).to(self.device)
    
    def prep_video_inputs(self, args, crop=None):
        # vid_path = r'C:\projects\VideoSketch\videos\production_id_4990428 (1080p).mp4'
        # out_dir = r'C:\projects\VideoSketch\preped_videos\ballerina'

        cap = cv2.VideoCapture(args.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert args.start_frame >= 0 and args.start_frame <= args.end_frame and args.end_frame < frame_count
        
        if not self.workdir.exists():
            makedirs(str(self.workdir))

        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        success, image = cap.read()
        is_first = True
        end_frame = args.end_frame + 1 if args.end_frame != -1 else args.end_frame
        for frame_index in range(args.start_frame, end_frame):
            self.prep_timer.tic()
            target, mask = self.process_image(image, args, crop, is_first)
            edges = self.calc_edges(target)
            is_first = False
            torch.save(target, str(self.workdir / f"edges_{frame_index}.t"))
            torch.save(mask, str(self.workdir / f"mask_{frame_index}.t"))
            torch.save(edges, str(self.workdir / f"frame_{frame_index}.t"))
            Image.fromarray(np.int8(edges.numpy() * 255)).save(str(self.workdir / f"edges_{frame_index}.png"))
            
            clip_attentions = self.clip_it(target)
            print('saving...', str(self.workdir / f"clip_attentions_{frame_index}.t"))
            torch.save(clip_attentions, str(self.workdir / f"clip_attentions_{frame_index}.t"))

            if self.attention_init:
                self.image_input_attn_clip = clip_attentions
                attention_map = self.set_attention_map()
                np.save(str(self.workdir / f"attention_map_{frame_index}.npy"), attention_map)
            else:
                self.attention_map = None
            
            success, image = cap.read()
            self.prep_timer.toc()
    
    def calc_edges(self, target):
        images = target.cpu().numpy()
        print(images.shape)
        images = np.transpose(np.transpose(images, (1, 2)), (2, 3))
        edges = np.zeros_like(images)
        for i, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, 100, 200)
            print(f'check canny max: {np.max(edge)}, min: {np.min(edge)}')
            if self.args.edges_blur:
                edge = cv2.GaussianBlur(edge, ksize=(self.args.edges_blur, self.args.edges_blur),
                                        sigmaX=0)
            edges[i] = np.clip(edge, 0, 1)
        return torch.from_numpy(np.transpose(np.transpose(edges, (3, 2)), (2, 1)))


    def dino_attn_helper(self, image):
        patch_size=8 # dino hyperparameter
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
            ])
        img = totens(image).to(self.device)
        self.w_featmap = img.shape[-2] // patch_size
        self.h_featmap = img.shape[-1] // patch_size

    def process_image(self, image, args, crop, is_first=False):
        if crop is not None:
            h_start, h_end, w_start, w_end = crop
            image = image[h_start:h_end, w_start:w_end]
        im_size = image.shape[:2]
        if max(im_size) > 512 and args.pre_resize == 0:
            args.pre_resize = 512
        if args.pre_resize > 0:
            resize_scale = args.pre_resize / max(im_size)
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
        if is_first:
            self.dino_attn_helper(target)
        
        masked_im, mask = utils.get_mask_u2net(args, target)
        
        if args.mask_object:
            target = masked_im
        if args.fix_scale:
            target = utils.fix_image_scale(target)

        transforms_ = []
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=Image.BICUBIC))
        assert args.image_scale >= args.center_crop
        center_crop = args.image_scale if args.center_crop == 0 else args.center_crop
        transforms_.append(transforms.CenterCrop(center_crop))
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

