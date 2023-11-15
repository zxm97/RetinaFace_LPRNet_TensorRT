# coding=utf-8

import os
import re
import time
import numpy as np
import cv2
from detect_lpd import LPDET
from detect_lpr import LPREC, cv2ImgAddText
import torch

if __name__ == '__main__':
    det = LPDET()
    rec = LPREC()

    # ----------------------------------------------------------
    if True:
        # img_path = 'E:/plate_recognition/CCPD2019/ccpd_np/'
        # img_path = 'E:/plate_recognition/license-plate-recoginition-main/test_images/'
        img_path = 'E:/plate_recognition/CCPD2020/ccpd_green/val/'
        # img_path = 'D:/license_plate_datasets/CRPD_all/CRPD_single/val/images/'
        # img_path = 'D:/license_plate_datasets/imgs_multi_line_yellow/'
        # img_path = 'D:/license_plate_datasets/imgs_single_line_white/'

        imgs = list(os.listdir(img_path))

        for img_ind, img_name in enumerate(imgs):
            # fp16 debug
            # if img_ind <2:
            #     continue
            time_start = time.time()
            # img_raw = cv2.imread(img_path+img_name, cv2.IMREAD_COLOR)
            img_raw = cv2.imdecode(np.fromfile(img_path+img_name, dtype=np.uint8), 1)

             # 1160, 720

            # resize_scale = 640 / min(img_raw.shape[0], img_raw.shape[1])
            # img_raw = cv2.resize(img_raw, (0, 0), fx=resize_scale, fy=resize_scale)
            # img_raw = cv2.resize(img_raw, (640, 640))
            is_det, processed_imgs, frame_full, bboxes_xyxy = det.det(img_raw)
            tic = time.time()
            for index, frame_crop in enumerate(processed_imgs):

                result = rec.rec(frame_crop)
                x1, y1 = bboxes_xyxy[index][0], bboxes_xyxy[index][1]
                frame_full = cv2ImgAddText(frame_full, result[0], (x1, y1-30))
                print(result)

                # cv2.imshow('frame_crop', frame_crop)
            time_used = time.time()-time_start
            print('fps=', 1/time_used)


            # cv2.imshow('frame_full', frame_full)
            # cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
    #

    # ----------------------------------------------------------------
    if False:
        total_time = 0
        for i in range(100):
            time_start = time.time()
            img_raw = cv2.imread('test_images/13.jpg', cv2.IMREAD_COLOR)
            # resize_scale = 640 / min(img_raw.shape[0], img_raw.shape[1])
            # img_raw = cv2.resize(img_raw, (0, 0), fx=resize_scale, fy=resize_scale)

            is_det, processed_imgs, frame_full, bboxes_xyxy = det.det(img_raw)
            if is_det:
                for frame_crop in processed_imgs:
                    # cv2.imshow('frame_crop', frame_crop)
                    tic = time.time()
                    result = rec.rec(frame_crop)
                    torch.cuda.synchronize() ############################################
                    print('rec net forward time: {:.4f}'.format(time.time() - tic))
                    print(result)
            # cv2.imshow('frame_full', frame_full)
            # cv2.waitKey(0)

            time_used = time.time()-time_start
            print('fps=', 1/time_used)
            total_time += time_used

            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
        print('total time:', total_time)

    # ----------------------------------------------------------------
    if False:
        cap = cv2.VideoCapture('test_videos/highway.mp4')
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            # resize_scale = 640 / min(frame.shape[0], frame.shape[1])
            # frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
            # frame = cv2.resize(frame, (640, 640))
            is_det, processed_imgs, frame_full,bboxes_xyxy= det.det(frame)

            # if cv2.waitKey(1000000) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()

            if is_det:
                print('is det')
                for frame_crop in processed_imgs:
                    time_start_rec = time.time()
                    result = rec.rec(frame_crop)
                    # torch.cuda.synchronize() #####################################################
                    print('rec net forward time: {:.4f}'.format(time.time() - time_start_rec))
                    #pattern = '^[京津沪冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川黔云渝藏陕陇青宁新闽粤晋琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}$'
                    pattern = '^([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[a-zA-Z](([DF]((?![IO])[a-zA-Z0-9](?![IO]))[0-9]{4})|([0-9]{5}[DF]))|[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1})$'
                    plate_number = result[0]
                    if re.match(pattern, plate_number):
                        print('有效车牌号',plate_number)
                    # else:
                    #     print('无效车牌号',plate_number)
                    # cv2.imshow('crop', frame_crop)

            cv2.imshow('full', frame_full)
            total_time = time.time()-t_start
            print('total time:', total_time)
            print('fps=', 1/total_time)
            print('-'*20)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
