# coding=utf-8
from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from retina_lpd.data import cfg_mnet, cfg_re50
from retina_lpd.layers.functions.prior_box import PriorBox
from retina_lpd.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from retina_lpd.models.retina import Retina
from retina_lpd.utils.box_utils import decode, decode_landm
import time
import torchvision
print(torch.__version__, torchvision.__version__)

parser = argparse.ArgumentParser(description='RetinaPL')
# 23 good
# parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_epoch_20_ccpd.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('-m', '--trained_model', default='E:/plate_recognition/license-plate-recoginition-main/retina_lpd/weights/mobilenet0.25_epoch_20_ccpd.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('-m', '--trained_model', default='retina_lpd/weights/ccpd_custom/mobilenet0.25_epoch_100_ccpd_blue_green_202310222226.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('-m', '--trained_model', default='retina_lpd/weights/ccpd_custom/mobilenet0.25_epoch_24_ccpd_blue+green+yellow+white_20231101.pth',
                    type=str, help='Trained state_dict file path to open')
# parser.add_argument('-m', '--trained_model', default='retina_lpd/weights/ccpd_custom/mobilenet0.25_epoch_99_custom_ccpd_resize640.pth',
#                     type=str, help='Trained state_dict file path to open')


parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold') ##### default=0.02
parser.add_argument('--top_k', default=1000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=500, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold') # 0.5
parser.add_argument('-image', default='test_images/0.jpg', help='test image path')
parser.add_argument('--draw', default=True, type=bool, help='visualization_threshold')#############################
parser.add_argument('--half', default=False, type=bool, help='use fp16') ######################
args = parser.parse_args()


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

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
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img



def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


# --------------------------------------------
from pyclbr import Function
from typing import Sequence

Tensor_list = dict()

def fp16_check(module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
    if isinstance(input, dict):
        for _, value in input.items():
            fp16_check(module, value, output)
        return
    if isinstance(input, Sequence):
        for value in input:
            fp16_check(module, value, output)
        return
    if isinstance(output, dict):
        for _, value in output.items():
            fp16_check(module, input, value)
        return
    if isinstance(output, Sequence):
        for value in output:
            fp16_check(module, input, value)
        return
    # with open('fp16.txt', 'a', encoding='utf-8') as f:
    #     f.write(str(module))
    #     f.write('\n')
    #     f.write(str(torch.mean(torch.abs(input))))
    #     f.write('\n')
    #     f.write(str(torch.mean(torch.abs(output))))
    #     f.write('\n')
    Tensor_list[str(module)] = [input, output]
    # print(module)
    # print(torch.abs(input).max()) # 检查是否有溢出
    # print(torch.abs(output).max())

    # print(torch.mean(torch.abs(input)))
    # print(torch.mean(torch.abs(output)))
    # if torch.abs(input).max()<65504  and torch.abs(output).max()>65504:
    #     print('from: ', module.finspect_name)
    # if torch.abs(input).max()>65504  and torch.abs(output).max()<65504:
    #     print('to: ', module.finspect_name)
    return



from contextlib import contextmanager
class FInspect:
    module_names = ['model']
    handlers = []

    def hook_all_impl(cls, module: torch.nn.Module, hook_func: Function)-> None:
        for name, child in module.named_children():
            cls.module_names.append(name)
            cls.hook_all_impl(cls, module=child, hook_func=hook_func)
        linked_name='->'.join(cls.module_names)
        setattr(module, 'finspect_name', linked_name)
        cls.module_names.pop()
        handler = module.register_forward_hook(hook=hook_func)
        cls.handlers.append(handler)

    @classmethod
    @contextmanager
    def hook_all(cls, module: torch.nn.Module, hook_func: Function)-> None:
        cls.hook_all_impl(cls, module, hook_func)
        yield
        [i.remove() for i in cls.handlers]

# --------------------------------------------



class LPDET:
    def __init__(self):
        torch.set_grad_enabled(False)
        self.cfg = None
        if args.network == "mobile0.25":
            self.cfg = cfg_mnet
        elif args.network == "resnet50":
            self.cfg = cfg_re50
        # net and model
        self.net = Retina(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, args.trained_model, args.cpu)

        self.net.eval()

        print('Finished loading detect model!')
        # print(self.net)
        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = self.net.to(self.device)

        if args.half:##################
            self.net.half()


        self.resize = 1

        self.im_height_last_frame = -1
        self.im_width_last_frame = -1

    def det(self, img_raw):
        # img_raw = cv2.imread(args.image, cv2.IMREAD_COLOR)
        tic = time.time()
        img_raw = resizeAndPad(img_raw, (640, 640))
        img = np.float32(img_raw)  # H, W, C

        img_origin = img_raw.copy()

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        ######### cpu numpy
        img -= (104, 117, 123) # rgb_mean = (104, 117, 123)  # bgr order
        img = img.transpose(2, 0, 1) # C, H, W
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        print('img Tensor shape:', img.shape) # [1, 3, 684, 1258]

        ######### gpu tensor
        # img = torch.from_numpy(img).to(self.device)
        # if args.half:
        #     img = img.half()
        # img -= (torch.Tensor((104, 117, 123)).to(self.device))
        #
        # img = img.permute(2, 0, 1)
        # img = img.unsqueeze(0)


        scale = scale.to(self.device)

        print('det net preprocess time: {:.4f}'.format(time.time() - tic))

        tic = time.time()
        with torch.no_grad():
            loc, conf, landms = self.net(img)  # forward pass
            # print(conf)
        print(loc.shape)
        print(conf.shape)
        print(landms.shape)
        # torch.Size([1, 16800, 4])
        # torch.Size([1, 16800, 2])
        # torch.Size([1, 16800, 8])

        # 查看网络输入输出值
        # with FInspect.hook_all(self.net, fp16_check):
        #     loc, conf, landms = self.net(img)
        # torch.save(Tensor_list, 'fp16.pt')

        torch.cuda.synchronize() #####################################################
        print('det net forward time: {:.4f}'.format(time.time() - tic))

        tic = time.time()
        # self.im_height_last_frame = 0
        # self.im_width_last_frame = 0 ####################################
        if self.im_height_last_frame != im_height or self.im_width_last_frame != im_width:
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            # torch.cuda.synchronize() #####################################################
            priors = priors.to(self.device)
            self.prior_data = priors.data

            print('priorBox time: {:.4f}'.format(time.time() - tic))
        self.im_height_last_frame = im_height
        self.im_width_last_frame = im_width

        tic = time.time()
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # print('decode time: {:.4f}'.format(time.time() - tic))

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

        # tic = time.time()
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold) ######################
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # print('nms time: {:.4f}'.format(time.time() - tic))


        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        print('postprocess time: {:.4f}'.format(time.time() - tic))

        is_det = False #####没检测到目标的处理
        # show image
        processed_imgs = []
        bboxes = []

        tic = time.time()
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                is_det = True #####
                text = "{:.4f}".format(b[4])
                # print(text)
                b = list(map(int, b))
                if args.draw:
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                if args.draw:
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                    # landms
                    cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                    cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                    # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                    cv2.circle(img_raw, (b[9], b[10]), 1, (0, 255, 0), 4)
                    cv2.circle(img_raw, (b[11], b[12]), 1, (255, 0, 0), 4)

                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                x1 = max(0, x1)
                x2 = min(x2, img_origin.shape[1])
                y1 = max(0, y1)
                y2 = min(y2, img_origin.shape[0])

                w = int(x2 - x1 + 1.0)
                h = int(y2 - y1 + 1.0)
                # img_box = np.zeros((h, w, 3))
                print(x1, y1, x2, y2)

                img_box = img_origin[y1:y2 + 1, x1:x2 + 1, :]
                # cv2.imshow("img_box",img_box)
                # cv2.waitKey(0)
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

                processed_imgs.append(processed)
                bboxes.append([b[0], b[1], b[2], b[3]])
                # 显示原图和处理后的图像
                # cv2.imshow("processed", processed)
                # save image

                name = "test.jpg"
                # cv2.imwrite(name, processed)
            # cv2.imshow('image', img_raw)
        print('align time: {:.4f}'.format(time.time() - tic))

        if is_det:
            return True, processed_imgs, img_raw, bboxes
        else:
            return False, processed_imgs, img_raw, bboxes



if __name__ == '__main__':
    
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = Retina(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    resize = 1

    # testing begin
    for i in range(1):
        
        img_raw = cv2.imread(args.image, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        tic = time.time()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()



        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

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
        print('priorBox+decode+nms time: {:.4f}'.format(time.time() - tic))
        # show image
        if args.save_image:
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
                
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                w = int(x2 - x1 + 1.0)
                h = int(y2 - y1 + 1.0)
                img_box = np.zeros((h, w, 3))
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
                
                # 显示原图和处理后的图像
                cv2.imshow("processed", processed)  
                # save image

                name = "test.jpg"
                cv2.imwrite(name, processed)
            cv2.imshow('image', img_raw)
            if cv2.waitKey(1000000) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

