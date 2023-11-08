import os
import cv2
import numpy as np
# import cupy as cp
import argparse
import torch
import tensorrt as trt
import time
import common
from retina_lpd.data import cfg_mnet as cfg_lpd
from math import ceil
from itertools import product as product
from retina_lpd.utils.nms.py_cpu_nms import py_cpu_nms
TRT_LOGGER = trt.Logger()


from utils import cfg as cfg_lpr, load_config
config_lpr = './config/lpr.yml'
load_config(cfg_lpr, config_lpr)

# input_w = 720
# input_h = 1160

parser = argparse.ArgumentParser(description='RetinaFaceTensorRT')
parser.add_argument('--lpd_engine_file_path', default='retina_lpd/weights/mobilenet0.25_epoch_24_ccpd_blue+green+yellow+white_20231101_fp16.engine', help='trt engine file')
parser.add_argument('--lpr_engine_file_path', default='./workspace/lpr_20231102_all_types_fp16.engine', help='trt engine file')
parser.add_argument('--input_w', default=640, type=int, help='tensorrt input width') # 720
parser.add_argument('--input_h', default=640, type=int, help='tensorrt input height') # 1160
parser.add_argument('--preprocess_method', default='numpy', type=str, help='numpy, torch, cupy')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
# parser.add_argument('-input_path', default='test_images/99999.jpg', help='test image path')
parser.add_argument('-input_path', default='E:/plate_recognition/CCPD2020/ccpd_green/val/', help='test image path')
# parser.add_argument('-input_path', default='./test_images/', help='test image path')
args = parser.parse_args()


def load_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("Engine file does not exist!")

def get_mean_matrix(mean_bgr):
    mean_np = np.array(mean_bgr).astype(np.float32)
    mean_np = np.expand_dims(mean_np, 0)
    mean_np = mean_np.repeat(args.input_w, axis=0)
    # print(mean_np)
    # print(mean_np.shape)
    mean_np = np.expand_dims(mean_np, 0)
    mean_np = mean_np.repeat(args.input_h, axis=0)
    return mean_np

def resizeAndPad(img, size, padColor=0):
    no_resize = False
    h, w = img.shape[:2]
    sh, sw = size

    if h == sh and w == sw:
        no_resize = True

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_LINEAR ####################### cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h
    if not no_resize:
        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        # set pad color
        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    else:
        scaled_img = img


    return scaled_img

def resizeAndPad_gpu(img, size, padColor=0):
    no_resize = False


    h, w = img.shape[:2]
    sh, sw = size

    if h == sh and w == sw:
        no_resize = True

    # interpolation method
    # if h > sh or w > sw: # shrinking image
    #     interp = cv2.INTER_AREA
    # else: # stretching image
    #     interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h
    if not no_resize:
    # compute scaling and pad sizing
        if aspect > sw/sh: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < sw/sh: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
    #     padColor = [padColor]*3

    # scale and pad
    # scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    # scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    img = np.float32(img)
    img = torch.from_numpy(img).cuda()
    # print(img)
    # mean = torch.Tensor((104, 117, 123)).cuda()
    # # print(mean)
    # img -= mean ############

    img = img.permute(2, 0, 1) # C, H, W

    img = img.unsqueeze(0) # 1 C, H, W


    # print('new_h, new_w', new_h, new_w)
    # print('pad_left, pad_right, pad_top, pad_bot', pad_left, pad_right, pad_top, pad_bot)
    if not no_resize:
        scaled_img = torch.nn.functional.interpolate(img, size=(new_h, new_w), scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)
        # pad(left, right, top, bottom)
        # print('scaled_img.shape after interpolating', scaled_img.shape)
        scaled_img = torch.nn.functional.pad(input=scaled_img, pad=(pad_left, pad_right, pad_top, pad_bot), mode='constant', value=padColor)
    else:
        scaled_img = img
    # print('scaled_img.shape after padding', scaled_img.shape)
    scaled_img = scaled_img.squeeze(0) # C, H, W


    scaled_img = scaled_img.permute(1, 2, 0) # H W C
    ret_img = scaled_img.clone() #  H W C
    mean = torch.Tensor([104, 117, 123]).cuda()
    # print(mean)
    ret_img = ret_img - mean  #  H W C
    # print('-mean',ret_img.shape) # [640, 640, 3]
    # print('ret_img', ret_img)
    # scaled_img= scaled_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # scaled_img = np.array(scaled_img, dtype=np.uint8)
    # cv2.imshow('asdasda', scaled_img)
    # cv2.waitKey(0)
    # print('scaled_img.shape', scaled_img.shape)
    ret_img = ret_img.permute(2, 0, 1) # H W C -> C H W
    scaled_img_ = scaled_img.cpu().numpy()#.astype(np.uint8)
    # del scaled_img
    scaled_img_ = np.ascontiguousarray(scaled_img_, dtype=np.uint8)
    img = ret_img.cpu().numpy()
    # del ret_img
    # print('scaled_img', scaled_img.shape)
    # print('ret_img', img.shape)
    # print('ret_img', ret_img)
    return scaled_img_, img


def preprocess_lpd(img_path, mean_matrix):
    tic = time.time()
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print('read img time', time.time()-tic)

    t_pre = time.time()

    # img_raw = cv2.resize(img_raw, (input_w, input_h))
    tic = time.time()
    img_raw  = resizeAndPad(img_raw, (args.input_h, args.input_w))
    print('letterbox resize time', time.time()-tic)
    # cv2.imshow('asda', img_raw)
    tic = time.time()
    img = img_raw.copy()
    img = np.float32(img)
    # img -= (104, 117, 123) # 0.0129 s
    img -= mean_matrix
    # print('-mean',img.shape) # (640, 640, 3)
    # img = img_raw - (104, 117, 123)
    print('subtract mean time', time.time()-tic)
    img = np.transpose(img, [2, 0, 1])
    # print(img_raw)
    # print(img)
    # print(img_raw[300:305, 300:305, :])
    # print(img[:, 300:310, 330:340])
        # CHW to NCHW format
    img = np.expand_dims(img, axis=0)
    # print('img.shape', img.shape) #(1, 3, 640, 640)
    # Convert the image to row-major order, also known as "C order":
    img = np.array(img, dtype=np.float32, order="C") # 0.006 s
    print('preprocess time(lpd): ', time.time()-t_pre)
    # print('img_raw.shape', img_raw.shape) # (640, 640, 3)
    # print('img.shape', img.shape) # (1, 3, 640, 640)



    return img_raw, img

def preprocess_gpu_lpd(img_path):
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    t_pre = time.time()

    ######### gpu tensor

    img_raw, img  = resizeAndPad_gpu(img_raw, (args.input_h, args.input_w))
    # print(img_raw.shape) # (3, 640, 640)
    # img  = resizeAndPad_gpu(img_raw.copy(), (500, 400))
    # img -= (torch.Tensor((104, 117, 123)).cuda())
    # img -= (torch.Tensor((123, 104, 117)).cuda())

    # print(img_raw[300:305, 300:305, :])
    # print(img[:, 300:310, 330:340])


    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32, order="C")
    # print('img_raw.shape', img_raw.shape) #(640, 640, 3)
    # print('img.shape', img.shape)# (1, 3, 640, 640)
    # print(img_raw)
    # print(img)
    #img_raw.shape should be (640, 640, 3)
    #'img.shape' should be (1, 3, 640, 640)
    # img_raw = np.transpose(img_raw, [1, 2, 0])
    # img_raw = np.array(img.copy(), dtype=np.uint8)
    # cv2.imshow('asd', img_raw)
    # cv2.waitKey(0)
    print('preprocess time(lpd): ', time.time()-t_pre)
    return img_raw, img



# def preprocess_cupy(img_path):
#     img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     t_pre = time.time()
#     img_raw = np.float32(img_raw)
#     ######### gpu tensor
#
#     img_raw  = resizeAndPad_cupy(img_raw, (args.input_h, args.input_w))
#     # img  = resizeAndPad_gpu(img_raw.copy(), (500, 400))
#     # img -= (torch.Tensor((104, 117, 123)).cuda())
#     # img -= (torch.Tensor((123, 104, 117)).cuda())
#
#     img = cp.asarray(img_raw.copy()) - cp.asarray((104, 117, 123))
#     # img = img_raw - (104, 117, 123)
#     img = cp.transpose(img, [2, 0, 1])
#         # CHW to NCHW format
#     img = cp.expand_dims(img, axis=0)
#     # Convert the image to row-major order, also known as "C order":
#     img = cp.asnumpy(img)
#     img = np.array(img, dtype=np.float32, order="C")
#     print('preprocess time:', time.time()-t_pre)
#     return img_raw, img

def preprocess_lpr(img):
    # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    tic = time.time()
    img = cv2.resize(img, tuple(cfg_lpr.input_size))

    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32, order="C")

    time_preprocess = time.time()-tic

    return img, time_preprocess


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    # boxes = torch.cat((
    #     priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    #     priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        #priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        ), 1)
    # landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
    #                     priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
    #                     #priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
    #                     priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
    #                     priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
    #                     ), dim=1)
    return landms

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes'] # [[16, 32], [64, 128], [256, 512]]
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        # output = torch.Tensor(anchors).view(-1, 4)
        # if self.clip:
        #     output.clamp_(max=1, min=0)
        output = np.reshape(anchors, (-1, 4))
        return output

def softmax_2d_np(input):
    input = np.exp(input- np.max(input))
    S = np.sum(input,axis=1)
    P = input / np.expand_dims(S, 1)
    return P

def postprocess_lpd(outputs, prior_data):

    loc, conf, landms = outputs

    loc = np.squeeze(loc, 0)
    conf = np.squeeze(conf, 0)
    conf = softmax_2d_np(conf)
    landms = np.squeeze(landms, 0)
    # should be
    # torch.Size([1, 16800, 4])
    # torch.Size([1, 16800, 2])
    # torch.Size([1, 16800, 8])

    # actual
    # (34372, 4)
    # (137488, 2)
    # (8593, 8)

    # print(loc.shape)
    # print(conf.shape)
    # print(landms.shape)

    resize = 1
    # scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = np.array([args.input_w, args.input_h, args.input_w, args.input_h])
    # print('scale.shape:',scale.shape)




    tic = time.time()
    # boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = decode(loc, prior_data, cfg_lpd['variance'])
    boxes = boxes * scale / resize
    print('boxes.shape', boxes.shape) # (16800, 4)
    # boxes = boxes.cpu().numpy()

    # scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    scores = conf[:, 1]
    # landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    landms = decode_landm(landms, prior_data, cfg_lpd['variance'])
    # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                        img.shape[3], img.shape[2],
    #                        img.shape[3], img.shape[2]])
    # scale1 = scale1.to(device)
    scale1 = np.array([args.input_w, args.input_h, args.input_w, args.input_h,
                           args.input_w, args.input_h,
                           args.input_w, args.input_h])
    # print('scale1.shape:',scale1.shape)
    landms = landms * scale1 / resize
    print('landms.shape', landms.shape) # (16800, 8)
    # landms = landms.cpu().numpy()

    print('decode time: {:.4f}'.format(time.time() - tic))


    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    tic =time.time()

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    print('nms time: {:.4f}'.format(time.time() - tic))

    return dets

def postprocess_lpr(outputs):
    prebs = outputs[0]
    preb_labels = list()
    for i in range(prebs.shape[0]):
        lpnum=''
        preb = prebs[i, :, :]
        # preb_label = torch.argmax(preb,dim=1)
        preb_label = np.argmax(preb,axis=1)
        # print()
        # print('predict before processing:', preb_label) ##############
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(cfg_lpr.CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        # print('preb_label', preb_label)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(cfg_lpr.CHARS) - 1):
                if c == len(cfg_lpr.CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
            # print('predict after processing:', no_repeat_blank_label) ###############

        for idx in no_repeat_blank_label:
            lpnum += cfg_lpr.CHARS[int(idx.item())]
        # print('lpnum', lpnum)
        preb_labels.append(lpnum)
    return preb_labels

def draw_bbox(img_raw, dets):
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        print(text)
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (255, 0, 0), 4)

    return img_raw

def crop_and_align_plates(img_raw, dets):
    plate_imgs = []
    for b in dets:
        print(b[4])
        if b[4] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        print(text)
        b = list(map(int, b))
        # cx = b[0]
        # cy = b[1] + 12

        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        print(x1, y1, x2, y2)
        # w = int(x2 - x1 + 1.0)
        # h = int(y2 - y1 + 1.0)
        # img_box = np.zeros((h, w, 3))
        img_box = img_raw[y1:y2 + 1, x1:x2 + 1, :]
        # cv2.imshow("img_box",img_box)
        # print('+++',b[9],b[10])

        new_x1, new_y1 = b[9] - x1, b[10] - y1
        new_x2, new_y2 = b[11] - x1, b[12] - y1
        new_x3, new_y3 = b[7] - x1, b[8] - y1
        new_x4, new_y4 = b[5] - x1, b[6] - y1
        # print(new_x1, new_y1)
        # print(new_x2, new_y2)
        # print(new_x3, new_y3)
        # print(new_x4, new_y4)

        # 定义对应的点
        points1 = np.float32([[new_x1, new_y1], [new_x2, new_y2], [new_x3, new_y3], [new_x4, new_y4]])
        points2 = np.float32([[0, 0], [94, 0], [0, 24], [94, 24]])

        # 计算得到转换矩阵
        M = cv2.getPerspectiveTransform(points1, points2)

        # 实现透视变换转换
        processed = cv2.warpPerspective(img_box, M, (94, 24))
        plate_imgs.append(processed)
    return plate_imgs

def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""

    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:


    # Download a dog image and save it to the following file path:
    input_path = args.input_path
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered

    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    # Load an image from the specified input path, and return it together with  a pre-processed version

    # Store the shape of the original input image in WH format, we will need it for later

    # Output shapes expected by the post-processor
    # feature map width or height = 640 / 32 = 20, 640 / 16 = 40, 640 / 8 = 80
    # number of anchors = 20**2 * 2 + 40**2 * 2 + 80**2 * 2 = 16800

    # num_of_anchors = int(((input_h / 32)**2 + (input_h / 16)**2 + (input_h / 8)**2) * 2)

    # num_of_anchors = int(((input_h / 32 * input_w / 32)
    #                       + (input_h / 16 * input_w / 16)
    #                       + (input_h / 8 * input_w / 8))
    #                      * 2)
    # output_shapes = [(1, num_of_anchors, 4), (1, num_of_anchors, 2), (1, num_of_anchors, 8)]
    output_shapes_lpd = [(1, -1, 4), (1, -1, 2), (1, -1, 8)]
    output_shapes_lpr = [(1, cfg_lpr.T_length, len(cfg_lpr.CHARS)),]
    # output_shapes = [(1, -1, 4), (-1, 2), (1, -1, 8)]
    # (67200,) -> [1, 16800, 4]
    # (33600,) -> [1, 16800, 2]
    # (134400,) -> [1, 16800, 8]

    priorbox = PriorBox(cfg_lpd, image_size=(args.input_h, args.input_w))
    priors = priorbox.forward()
    # priors = priors.to(device)
    # prior_data = priors.data
    prior_data = priors

    with load_engine(args.lpd_engine_file_path) as engine_lpd, engine_lpd.create_execution_context() as context_lpd:
        with load_engine(args.lpr_engine_file_path) as engine_lpr, engine_lpr.create_execution_context() as context_lpr:
            inputs_lpd, outputs_lpd, bindings_lpd, stream_lpd = common.allocate_buffers(engine_lpd)
            inputs_lpr, outputs_lpr, bindings_lpr, stream_lpr = common.allocate_buffers(engine_lpr)
            if args.preprocess_method == 'numpy':
                mean_matrix = get_mean_matrix([104, 117, 123])
            if os.path.isdir(input_path):
                print("Running inference on images...")
                image_files = os.listdir(input_path)
                for image_file in image_files:
                    time_start = time.time()
                    if args.preprocess_method == 'torch':
                        image_raw, image = preprocess_gpu_lpd(os.path.join(input_path, image_file))
                    elif args.preprocess_method == 'numpy':
                        image_raw, image = preprocess_lpd(os.path.join(input_path, image_file), mean_matrix)
                    # elif args.preprocess_method == 'cupy':
                    #     image_raw, image = preprocess_cupy(os.path.join(input_path, image_file))

                    tic = time.time()
                    inputs_lpd[0].host = image
                    trt_outputs_lpd = common.do_inference_v2(context_lpd, bindings=bindings_lpd, inputs=inputs_lpd, outputs=outputs_lpd, stream=stream_lpd)
                    trt_outputs_lpd = [output.reshape(shape) for output, shape in zip(trt_outputs_lpd, output_shapes_lpd)]
                    print('inference time(lpd): ', time.time()-tic)
                    tic = time.time()
                    dets = postprocess_lpd(trt_outputs_lpd, prior_data)
                    print('postprocess time(lpd):', time.time()-tic)
                    plate_imgs = crop_and_align_plates(image_raw, dets)
                    # for ind_plate, img_plate in enumerate(plate_imgs):
                    #     cv2.imshow('plate' + str(ind_plate+1), img_plate)
                    # image_raw = draw_bbox(image_raw, dets)
                    # cv2.imshow('image', image_raw)
                    # cv2.waitKey(0)
                    # cv2.waitKey(2000)
                    for ind_plate, img_plate in enumerate(plate_imgs):
                        img_plate, preprocess_time_lpr = preprocess_lpr(img_plate)
                        print('preprocess time (lpr):', preprocess_time_lpr)
                        tic = time.time()
                        inputs_lpr[0].host = img_plate
                        trt_outputs_lpr = common.do_inference_v2(context_lpr, bindings=bindings_lpr, inputs=inputs_lpr, outputs=outputs_lpr, stream=stream_lpr)
                        trt_outputs_lpr = [output.reshape(shape) for output, shape in zip(trt_outputs_lpr, output_shapes_lpr)]
                        print('inference time(lpr): ', time.time()-tic)
                        tic = time.time()
                        results = postprocess_lpr(trt_outputs_lpr)
                        print('postprocess time(lpr): ', time.time()-tic)
                        print(results)




                    time_end = time.time()
                    print('time total:', time_end - time_start)
                    print('fps:', 1 / (time_end - time_start))
                    print('*' * 20)
            else:
                if input_path.find('.jpg') != -1: # is image
                    print("Running inference on image {}...".format(input_path))
                    image_raw, image = preprocess_lpd(input_path)
                    inputs_lpd[0].host = image
                    trt_outputs_lpd = common.do_inference_v2(context_lpd, bindings=bindings_lpd, inputs=inputs_lpd, outputs=outputs_lpd, stream=stream_lpd)
                    trt_outputs_lpd = [output.reshape(shape) for output, shape in zip(trt_outputs_lpd, output_shapes_lpd)]
                    # print(len(trt_outputs))
                    # for trt_output in trt_outputs:
                    #     print(trt_output.shape)

                    dets = postprocess_lpd(trt_outputs_lpd, prior_data)

                    plate_imgs = crop_and_align_plates(image_raw, dets)
                    # for ind_plate, img_plate in enumerate(plate_imgs):
                    #     cv2.imshow('plate' + str(ind_plate+1), img_plate)
                    image_raw = draw_bbox(image_raw, dets)

                    for ind_plate, img_plate in enumerate(plate_imgs):
                        img_plate, preprocess_time_lpr = preprocess_lpr(img_plate)
                        print('preprocess time (lpr):', preprocess_time_lpr)
                        tic = time.time()
                        inputs_lpr[0].host = img_plate
                        trt_outputs_lpr = common.do_inference_v2(context_lpr, bindings=bindings_lpr, inputs=inputs_lpr, outputs=outputs_lpr, stream=stream_lpr)
                        trt_outputs_lpr = [output.reshape(shape) for output, shape in zip(trt_outputs_lpr, output_shapes_lpr)]
                        print('inference time(lpr): ', time.time()-tic)
                        tic = time.time()
                        results = postprocess_lpr(trt_outputs_lpr)
                        print('postprocess time(lpr): ', time.time()-tic)
                        print(results)


                    cv2.imshow('image', image_raw)
                    cv2.waitKey(0)



main()
