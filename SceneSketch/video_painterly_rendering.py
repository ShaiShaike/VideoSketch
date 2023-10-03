import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import sys
sys.stdout.flush()

import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange
from numpy.random import randint as nprandint
from pathlib import Path

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from models.video_painter import VideoPainter
from IPython.display import display, SVG
import matplotlib.pyplot as plt
# from torch import autograd


def load_renderer(args):
    renderer = VideoPainter(args=args, num_strokes=args.num_paths,
                            num_segments=args.num_segments,
                            imsize=args.image_scale,
                            device=args.device)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
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
        args.image_scale, interpolation=PIL.Image.BICUBIC))
    transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)

    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    mask = Image.fromarray((mask*255).astype(np.uint8)).convert('RGB')
    mask = data_transforms(mask).unsqueeze(0).to(args.device)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return target_, mask


def main(args):
    loss_func = Loss(args)
    # utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args)
    
    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss, best_num_strokes = 100, 100, args.num_paths
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-7
    terminate = False

    renderer.set_random_noise(0)
    renderer.init_image(stage=0)
    renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"init_svg") # this is the inital random strokes
    optimizer.init_optimizers()
    
    # not using tdqm for jupyter demo
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    if args.switch_loss:
        # start with width optim and than switch every switch_loss iterations
        renderer.turn_off_points_optim()
        optimizer.turn_off_points_optim()

    center_margins = max(args.center_frame - args.start_frame,
                         args.end_frame - args.center_frame)
    if 'centerloss' in args.center_method:
        renderer.load_clip_attentions_and_mask(args.center_frame)
        center_inputs = renderer.get_target(args.center_frame).detach()
        center_mask = renderer.mask
    print('1:', args.center_method, 'centerloss' in args.center_method)
    with torch.no_grad():
        if 'centerloss' in args.center_method:
            init_sketches, init_motions, _ = (tensor.to(args.device) for tensor in renderer.get_image("init"))
        else:  
            init_sketches, init_motions = (tensor.to(args.device) for tensor in renderer.get_image("init"))
        renderer.save_svg(
                f"{args.output_dir}", f"init")
    
    center_weight = 0.5

    for epoch in epoch_range:
        # Todo: args.batch_size
        if 'centeriter' in args.center_method:
            epoch_margins = min(int(epoch * args.center_interval_ratio / args.num_iter * center_margins), center_margins)
            batch_frame_indexes = nprandint(
                max(args.center_frame - epoch_margins, args.start_frame),
                min(args.center_frame + epoch_margins, args.end_frame) + 1)
        else:
            batch_frame_indexes = nprandint(args.start_frame, args.end_frame + 1)
        
        if not args.display:
            epoch_range.refresh()
        start = time.time()
        optimizer.zero_grad_()
        renderer.load_clip_attentions_and_mask(batch_frame_indexes)
        inputs = renderer.get_target(batch_frame_indexes)
        if 'centerloss' in args.center_method:
            sketches, motions, center_sketches = (tensor.to(args.device) for tensor in renderer.get_image())
            print("sketches", sketches.size(), "center_sketches", center_sketches.size())
            print("inputs", inputs.size(), "center_inputs", center_inputs.size())
            center_losses_dict_weighted, _, _ = loss_func(
                center_sketches, center_inputs, counter, renderer.get_widths(), renderer, optimizer,
                mode="train", width_opt=renderer.width_optim, mask=center_mask)
        else:
            sketches, motions = (tensor.to(args.device) for tensor in renderer.get_image())
        losses_dict_weighted, losses_dict_norm, losses_dict_original = loss_func(
            sketches, inputs.detach(), counter, renderer.get_widths(), renderer, optimizer,
            mode="train", width_opt=renderer.width_optim)
        motion_regularization = motions[:, 1:] - motions[:, :-1]
        motion_regularization = motion_regularization * motion_regularization
        args.motion_reg_ratio *= 0.998
        center_weight = np.sin(np.pi * epoch / args.num_iter)
        loss = sum(list(losses_dict_weighted.values())) + args.motion_reg_ratio * torch.sum(motion_regularization)
        if 'centerloss' in args.center_method:
            loss += center_weight * sum(list(center_losses_dict_weighted.values()))
        loss.backward()
        optimizer.step_()

        if epoch % args.save_interval == 0:
            # utils.plot_batch(inputs, sketches, f"{args.output_dir}/jpg_logs", counter,
            #                  use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")

        if epoch % args.eval_interval == 0 and epoch >= args.min_eval_iter:
            if args.width_optim:
                if args.mlp_train and args.optimize_points:
                    torch.save({
                        'model_state_dict': renderer.get_mlp().state_dict(),
                        'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                        }, f"{args.output_dir}/mlps/points_mlp{counter}.pt")
                    torch.save({
                        'model_state_dict': renderer.get_motion_mlp().state_dict(),
                        'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                        }, f"{args.output_dir}/mlps/motion_mlp{counter}.pt")
                torch.save({
                    'model_state_dict': renderer.get_width_mlp().state_dict(),
                    'optimizer_state_dict': optimizer.get_width_optim().state_dict(),
                    }, f"{args.output_dir}/mlps/width_mlp{counter}.pt")

            with torch.no_grad():
                #Todo: evaluate by last frame or by first, middle last
                renderer.load_clip_attentions_and_mask(args.end_frame)
                inputs = renderer.get_target(args.end_frame)
                if 'centerloss' in args.center_method:
                    sketches, motions, center_sketches = (tensor.to(args.device) for tensor in renderer.get_image())
                else:
                    sketches, motions = (tensor.to(args.device) for tensor in renderer.get_image())
                losses_dict_weighted_eval, losses_dict_norm_eval, losses_dict_original_eval = loss_func(sketches, inputs, counter, renderer.get_widths(), renderer=renderer, mode="eval", width_opt=renderer.width_optim)
                motion_regularization_eval = motions[:, 1:] - motions[:, :-1]
                motion_regularization_eval = motion_regularization_eval * motion_regularization_eval
                loss_eval = sum(list(losses_dict_weighted_eval.values())) #+ args.motion_reg_ratio * sum(motion_regularization_eval)
                configs_to_save["loss_eval"].append(loss_eval.item())
                if "num_strokes" not in configs_to_save.keys():
                    configs_to_save["num_strokes"] = []
                configs_to_save["num_strokes"].append(renderer.get_strokes_count())
                for k in losses_dict_norm_eval.keys():
                    original_name, gradnorm_name, final_name = k + "_original_eval", k + "_gradnorm_eval", k + "_final_eval"
                    if original_name not in configs_to_save.keys():
                        configs_to_save[original_name] = []
                    if gradnorm_name not in configs_to_save.keys():
                        configs_to_save[gradnorm_name] = []
                    if final_name not in configs_to_save.keys():
                        configs_to_save[final_name] = []
                    
                    configs_to_save[original_name].append(losses_dict_original_eval[k].item())
                    configs_to_save[gradnorm_name].append(losses_dict_norm_eval[k].item())
                    if k in losses_dict_weighted_eval.keys():
                        configs_to_save[final_name].append(losses_dict_weighted_eval[k].item())                

                cur_delta = loss_eval.item() - best_loss
                print(f"epoch: {epoch}: total loss: {loss_eval.item()}, lr: {optimizer.scheduler.get_lr()},",
                      f"motion weight: {args.motion_reg_ratio}, center_weight: {center_weight}")
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        best_num_strokes = renderer.get_strokes_count()
                        terminate = False
                        
                        if args.mlp_train and args.optimize_points and not args.width_optim:
                            torch.save({
                                'model_state_dict': renderer.get_mlp().state_dict(),
                                'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                                }, f"{args.output_dir}/points_mlp.pt")
                            torch.save({
                                'model_state_dict': renderer.get_motion_mlp().state_dict(),
                                'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                                }, f"{args.output_dir}/motion_mlp.pt")
                """     
                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb.run.summary["best_num_strokes"] = best_num_strokes
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_original_eval.keys():
                        wandb_dict[k + "_original_eval"] = losses_dict_original_eval[k].item()                    
                    for k in losses_dict_norm_eval.keys():
                        wandb_dict[k + "_gradnorm_eval"] = losses_dict_norm_eval[k].item()
                    for k in losses_dict_weighted_eval.keys():
                        wandb_dict[k + "_final_eval"] = losses_dict_weighted_eval[k].item()
                    wandb.log(wandb_dict, step=counter)
                """
        """
        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            if args.width_optim:
                wandb_dict["lr_width"] = optimizer.get_lr("width")
                wandb_dict["num_strokes"] = renderer.get_strokes_count()
            # wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr(), "num_strokes": optimizer.}
            for k in losses_dict_original.keys():
                wandb_dict[k + "_original"] = losses_dict_original[k].item()
            for k in losses_dict_norm.keys():
                wandb_dict[k + "_gradnorm"] = losses_dict_norm[k].item()
            for k in losses_dict_weighted.keys():
                wandb_dict[k + "_final"] = losses_dict_weighted[k].item()
            
            wandb.log(wandb_dict, step=counter)
        """
        counter += 1
        if args.switch_loss:
            if epoch > 0 and epoch % args.switch_loss == 0:
                    renderer.switch_opt()
                    optimizer.switch_opt()
    if args.width_optim:
        utils.log_best_normalised_sketch(configs_to_save, args.output_dir, args.use_wandb, args.device, args.eval_interval, args.min_eval_iter)
    utils.inference_video(args)
    return configs_to_save

if __name__ == "__main__":
    args = config.parse_video_arguments()
    assert Path(args.workdir).exists()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
