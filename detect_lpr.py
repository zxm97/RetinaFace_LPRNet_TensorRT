# coding=utf-8
'''
Date: 2022-01-11 21:45:55
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-03-02 21:49:11
FilePath: /license-plate-recoginition/inference.py
'''

import argparse
import os
import sys
import time
from locale import normalize
sys.path.insert(0,os.getcwd())
import cv2
import numpy as np
# import pytorch_lightning as pl
import torch
# from pytorch_lightning.callbacks import ProgressBar
# from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from PIL import Image, ImageDraw, ImageFont

#from data.load_data import LPRDataLoader, collate_fn
from task import TrainingTask
from utils import (LPLightningLogger, cfg, get_image_list, load_config,
                   load_model_weight, mkdir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="val", help="task to run, test or val")
    # parser.add_argument("--config", type=str, help="model config file(.yml) path")
    # parser.add_argument("--model", type=str, help="ckeckpoint file(.ckpt) path")
    # parser.add_argument("--path", type=str, help="img path")
    # parser.add_argument("--config", type=str, default="config/lpr-0213.yml", help="model config file(.yml) path")
    # parser.add_argument("--model", type=str,default="workspace/lp_model_best.pth",help="ckeckpoint file(.ckpt) path")
    parser.add_argument("--config", type=str, default="config/lpr.yml", help="model config file(.yml) path")
    parser.add_argument("--model", type=str,default="workspace/lpr_20231102_all_types.ckpt",help="ckeckpoint file(.ckpt) path")

    # parser.add_argument("--path", type=str, help="img path")
    parser.add_argument("--path", type=str, default='E:/plate_recognition/CBLPRD-330k_v1/val_yellow',help="img path")
    parser.add_argument('--half', default=False, type=bool, help='use fp16') ######################
    args = parser.parse_args()
    return args

args = parse_args()

class Perdictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        logger.info("Creating model...")
        self.task = TrainingTask(cfg)
        ckpt = torch.load(model_path)
        # self.task.model.load_state_dict(ckpt["state_dict"])
        self.task.model.load_state_dict({key.replace('model.', ''): value for key, value in ckpt["state_dict"].items()})
        self.task.model.eval() #############这行没加所以结果精度很低
        self.task.model.to(device)
        if args.half:
            self.task.model.half()
        self.device = device

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1) ############
            #img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        tic = time.time()
        meta["img"] = self.preprocess(meta["img"])
        meta["img"] = (
            torch.from_numpy(meta["img"].transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        print('preprocess time:',time.time() - tic)
        with torch.no_grad():
            results = self.task.inference(meta)

        return meta, results

    def inference_2(self, img): # 输入是图片而不是文件名
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            # img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta["img"] = self.preprocess(meta["img"])
        # print('meta["img"].shape', meta["img"].shape)
        meta["img"] = (
            torch.from_numpy(meta["img"].transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        if args.half:
            meta["img"] = meta["img"].half()
        with torch.no_grad():
            results = self.task.inference(meta)
        return meta, results

    def preprocess(self, img):
        img = cv2.resize(img, tuple(self.cfg.input_size))
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        return img

def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    device='cpu' 
    if torch.cuda.is_available():
        device='cuda' 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    mkdir(local_rank, cfg.save_dir)
    logger = LPLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)
    LP_Rec = Perdictor(cfg, args.model, logger, device=device)

    if os.path.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    count_correct = 0
    for image_name in files:
        meta, result = LP_Rec.inference(image_name)
        # lp_number = meta["img_info"]["file_name"].split('-')[2] ###########
        #
        # target = []
        # for chara in lp_number:
        #     target.append(cfg.CHARS.index(chara))
        # print('target:', target)
        # img = meta["raw_img"]
        # save_path = 'workspace/result/' + meta["img_info"]["file_name"]
        # img = cv2ImgAddText(img, result[0], (0, 0))
        # cv2.imwrite(save_path, img)
        # print(f'file_name:{meta["img_info"]["file_name"]},pred:{result[0]}')
        # target = meta["img_info"]["file_name"].split('-')[2]
        # if target!=result[0]:
        #     save_path = 'workspace/result/' + meta["img_info"]["file_name"]
        #     img = meta["raw_img"]
        #     img = cv2ImgAddText(img, result[0], (0, 0))
        #     cv2.imwrite(save_path, img)
        #     print(f'target:{target},pred:{result[0]}')


        # if lp_number == result[0]:
        #     count_correct += 1
    print('acc:', count_correct / (len(files)))

class LPREC:
    def __init__(self):
        load_config(cfg, args.config)
        local_rank = -1
        device='cpu'
        if torch.cuda.is_available():
            device='cuda'
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        mkdir(local_rank, cfg.save_dir)
        logger = LPLightningLogger(cfg.save_dir)
        logger.dump_cfg(cfg)
        self.LP_Rec = Perdictor(cfg, args.model, logger, device=device)

    def rec(self, img):
        meta, result = self.LP_Rec.inference_2(img)
        # print(result)
        return result

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    main(args)
