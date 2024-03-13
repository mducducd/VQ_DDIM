import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from tqdm import tqdm
import logging
from dataset import ReactionDataset, TestDataset, get_test_dataloader
from model import TransformerVAE
from utils import AverageMeter
from render import Render
from model.losses import VAELoss
from metric import *
from dataset import get_dataloader, get_test_dataloader
from utils import load_config
import model as module_arch
import model.losses as module_loss
from functools import partial

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Param
    parser.add_argument('--dataset-path', default="./data", type=str, help="dataset path")
    parser.add_argument('--split', type=str, help="split of dataset", choices=["val", "test"], required=True)
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-j', '--num_workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--img-size', default=256, type=int, help="size of train/test image data")
    parser.add_argument('--crop-size', default=224, type=int, help="crop size of train/test image data")
    parser.add_argument('-max-seq-len', default=751, type=int, help="max length of clip")
    parser.add_argument('--clip-length', default=751, type=int, help="len of video clip")
    parser.add_argument('--window-size', default=8, type=int, help="prediction window-size for online mode")
    parser.add_argument('--feature-dim', default=128, type=int, help="feature dim of model")
    parser.add_argument('--audio-dim', default=78, type=int, help="feature dim of audio")
    parser.add_argument('--_3dmm-dim', default=58, type=int, help="feature dim of 3dmm")
    parser.add_argument('--emotion-dim', default=25, type=int, help="feature dim of emotion")
    parser.add_argument('--online', action='store_true', help='online / offline method')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")
    parser.add_argument('--device', default='cuda', type=str, help="device: cuda / cpu")
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--threads', default=8, type=int, help="num max of threads")
    parser.add_argument('--binarize', action='store_true', help='binarize AUs output from model')

    args = parser.parse_args()
    return args

# Evaluating
def val(args, model, val_loader, criterion, render, binarize=False):
    losses = AverageMeter()
    rec_losses = AverageMeter()
    kld_losses = AverageMeter()
    model.eval()

    out_dir = os.path.join(args.outdir, args.split)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    listener_emotion_gt_list = []
    listener_emotion_pred_list = []
    speaker_emotion_list = []
    all_listener_emotion_pred_list = []

    for batch_idx, (speaker_emotion) in enumerate(tqdm(val_loader)):
        if torch.cuda.is_available():
            # print('SHAPE: ', speaker_video_clip.shape, speaker_audio_clip.shape, speaker_emotion.shape, listener_emotion.shape, listener_3dmm.shape, listener_references.shape)
            # if len(speaker_video_clip.shape) != 1: # if loaded
            #     speaker_video_clip, speaker_audio_clip = speaker_video_clip[:,:750].cuda(), speaker_audio_clip[:,:750].cuda()
            speaker_emotion= speaker_emotion[:,:750].cuda()
        with torch.no_grad():
            prediction = model(speaker_emotion=speaker_emotion)

            # if isinstance(prediction, list): # Trans VAE
            #     listener_3dmm_out, listener_emotion_out, distribution = prediction
            #     loss, rec_loss, kld_loss = criterion(listener_emotion, listener_3dmm, listener_emotion_out, listener_3dmm_out, distribution)

            #     losses.update(loss.data.item(), speaker_video_clip.size(0))
            #     rec_losses.update(rec_loss.data.item(), speaker_video_clip.size(0))
            #     kld_losses.update(kld_loss.data.item(), speaker_video_clip.size(0))
            # else: # BeLFusion
            listener_3dmm_out = prediction["3dmm_coeff"]
            listener_emotion_out = prediction["prediction"]
            loss = criterion(**prediction)["loss"].item()
            losses.update(loss)
            
            # binarize first 15 positions
            if binarize:
                listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
            B = speaker_video_clip.shape[0]
            # if (batch_idx % 25) == 0:
            #     for bs in range(B):
            #         render.rendering_for_fid(out_dir, "{}_b{}_ind{}".format(args.split, str(batch_idx + 1), str(bs + 1)),
            #                 listener_3dmm_out[bs], speaker_video_clip[bs], listener_references[bs], listener_video_clip[bs,:750])
            listener_emotion_pred_list.append(listener_emotion_out.cpu())

            speaker_emotion_list.append(speaker_emotion.cpu())

    listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim = 0)

    all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))

    print("-----------------Repeat 9 times-----------------")
    for i in range(9):
        listener_emotion_pred_list = []
        for batch_idx, (
        speaker_video_clip, speaker_audio_clip, speaker_emotion, _, _, _, listener_emotion, _, _) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                if len(speaker_video_clip.shape) != 1: # if loaded
                    speaker_video_clip, speaker_audio_clip = \
                        speaker_video_clip[:,:750].cuda(), speaker_audio_clip[:,:750].cuda()
                speaker_emotion, listener_emotion = speaker_emotion[:,:750].cuda(), listener_emotion[:,:750].cuda()
            with torch.no_grad():
                prediction = model(speaker_video=speaker_video_clip, speaker_audio=speaker_audio_clip, speaker_emotion=speaker_emotion, listener_emotion=listener_emotion)
                if isinstance(prediction, list): # Trans VAE
                    listener_emotion_out = prediction[1]
                else: # BeLFusion
                    listener_emotion_out = prediction["prediction"]

                # binarize first 15 positions
                if binarize:
                    listener_emotion_out[:, :, :15] = torch.round(listener_emotion_out[:, :, :15])
                listener_emotion_pred_list.append(listener_emotion_out.cpu())
        listener_emotion_pred = torch.cat(listener_emotion_pred_list, dim=0)
        all_listener_emotion_pred_list.append(listener_emotion_pred.unsqueeze(1))
        
    all_listener_emotion_pred = torch.cat(all_listener_emotion_pred_list, dim=1)

    print("-----------------Evaluating Metric-----------------")

    p = args.threads
    
    FRDvs = compute_FRDvs(all_listener_emotion_pred)
    FRVar  = compute_FRVar(all_listener_emotion_pred)
    smse  = compute_s_mse(all_listener_emotion_pred)

    return FRDvs, FRVar, smse



def main(args):
    checkpoint_path = args.resume
    
    config_path = os.path.join(os.path.dirname(checkpoint_path.replace('checkpoint_99.pth', '')), "config.yaml")

    cfg = load_config(config_path)
    dataset_cfg = cfg.validation_dataset if args.split == "val" else cfg.test_dataset
    dataset_cfg.dataset_path = args.dataset_path
    # dataset_cfg.num_workers = 1
    # dataset_cfg.batch_size = 16
    print('workers',dataset_cfg.num_workers)
    val_loader = get_test_dataloader(dataset_cfg, args.split, load_emotion_s=True)
    # val_loader = get_dataloader(args, args.split, load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=True, load_emotion_l=True, load_3dmm_l=True, load_ref=True)
    model = getattr(module_arch, cfg.arch.type)(cfg.arch.args)
    criterion = partial(getattr(module_loss, cfg.loss.type), **cfg.loss.args)

    if args.resume != '': #  resume from a checkpoint
        print("Resume from {}".format(checkpoint_path))
        checkpoints = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoints['state_dict']
        model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        render = Render('cuda')
    else:
        render = Render()

    FRDvs, FRVar, smse = val(args, model, val_loader, criterion, render, binarize=args.binarize)
   
    print("Metric: | S-MSE: {:.5f} | FRVar: {:.5f} | FRDvs: {:.5f} ".format(smse, FRVar, FRDvs))



# ---------------------------------------------------------------------------------


if __name__=="__main__":
    args = parse_arg()
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    main(args)