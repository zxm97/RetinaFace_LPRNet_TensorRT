import os
import cv2
import numpy as np
import argparse
import tensorrt as trt
import time
import common
TRT_LOGGER = trt.Logger()

from utils import cfg, load_config
config = './config/lpr.yml'
load_config(cfg, config)

parser = argparse.ArgumentParser(description='LPRNet_MultiLines_TensorRT')
parser.add_argument('--engine_file_path', default='./workspace/lpr_20231102_all_types_fp16.engine', help='trt engine file')
# parser.add_argument('--engine_file_path', default='./workspace/lpr_20231102_all_types_int8.engine', help='trt engine file')
parser.add_argument('--preprocess_method', default='numpy', type=str, help=' ')
# parser.add_argument('-input_path', default='test_images/99999.jpg', help='test image path')
parser.add_argument('-input_path', default='E:/plate_recognition/CBLPRD-330k_v1/val_yellow', help='test image path')
parser.add_argument('--show_plate', default=False, type=bool, help='show plate images before doing inference')
parser.add_argument('--print_info', default=False, type=bool, help='print recognition result and time used')
args = parser.parse_args()


def load_engine(engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        print("Engine file does not exist!")


def preprocess(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    tic = time.time()
    img = cv2.resize(img, tuple(cfg.input_size))
    if args.show_plate:
        cv2.imshow('plate', img)
        cv2.waitKey(0)
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32, order="C")

    time_preprocess = time.time()-tic
    if args.print_info:
        print('preprocess time:', time_preprocess)
    return img, time_preprocess


def postprocess(outputs):
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
        if pre_c != len(cfg.CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        # print('preb_label', preb_label)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(cfg.CHARS) - 1):
                if c == len(cfg.CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
            # print('predict after processing:', no_repeat_blank_label) ###############

        for idx in no_repeat_blank_label:
            lpnum += cfg.CHARS[int(idx.item())]
        # print('lpnum', lpnum)
        preb_labels.append(lpnum)
    return preb_labels



def main():
    input_path = args.input_path
    output_shapes = [(1, cfg.T_length, len(cfg.CHARS)),]
    with load_engine(args.engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        if os.path.isdir(input_path):
            print("Running inference on images...")
            image_files = os.listdir(input_path)
            t_preprocess_total = 0
            t_inference_total = 0
            t_postprocess_total= 0
            t_total = 0
            for image_file in image_files:
                time_start = time.time()
                if args.preprocess_method == 'torch':
                    # not implemented
                    image, time_preprocess = preprocess(os.path.join(input_path, image_file))
                elif args.preprocess_method == 'numpy':
                    image, time_preprocess = preprocess(os.path.join(input_path, image_file))
                t_preprocess_total += time_preprocess
                tic = time.time()
                inputs[0].host = image
                trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
                time_inference = time.time()-tic
                if args.print_info:
                    print('inference time:', time_inference)
                t_inference_total += time_inference
                tic = time.time()
                results = postprocess(trt_outputs)
                time_postprocess = time.time()-tic
                if args.print_info:
                    print('postprocess time:', time_postprocess)
                t_postprocess_total += time_postprocess
                if args.print_info:
                    print(results)

                time_total = time.time() - time_start
                t_total += time_total
                if args.print_info:
                    print('time total:', time_total)
                    print('fps:', 1 / (time_total))
            print('run trt model on ' + str(len(image_files)) + ' images')
            print('preprocess time (average) = ', t_preprocess_total / len(image_files))
            print('inference time (average) = ', t_inference_total / len(image_files))
            print('postprocess time (average) = ', t_postprocess_total / len(image_files))
            print('all time (average) = ', t_total / len(image_files)) # read image + preprocess + inference + postprocess

        else:
            if input_path.find('.jpg') != -1: # is image
                print("Running inference on image {}...".format(input_path))
                image, _ = preprocess(input_path)
                inputs[0].host = image
                trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
                results = postprocess(trt_outputs)
                print(results)
                cv2.imshow('image', image)
                cv2.waitKey(0)

if __name__ == '__main__':
    main()
