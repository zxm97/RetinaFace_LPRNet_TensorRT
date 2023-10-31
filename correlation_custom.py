import cv2
import numpy as np
import time
# 获取最可疑区域轮廓点集
import skimage.filters as filters



def correlation(x1, y1, x2, y2, img):
    tic = time.time()
    x = x1
    y = y1
    w = (x2 - x1)
    h = (y2 - y1)

    x -= w * 0.14
    w += w * 0.28
    y -= h * 0.4
    h += h * 0.8
    img = img.copy()
    img = img[int(y):int(y+h), int(x):int(x+w), :]
    cv2.imshow('cropped',img)
    cv2.waitKey(0)
    blocksize=15
    C=-10

    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # binary_niblack = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,C) #邻域大小17是不是太大了??

    # blur
    smooth = cv2.GaussianBlur(gray_image, (5,5), 0)

    smooth = cv2.Canny(smooth, 75, 200)
    cv2.imshow("image",smooth)
    cv2.waitKey(0)
    # divide gray by morphology image
    division = cv2.divide(gray_image, smooth, scale=255)


    # sharpen using unsharp masking
    sharp = filters.unsharp_mask(division, radius=1.5, amount=1.5, multichannel=False, preserve_range=False)
    gray_image = (255*sharp).clip(0,255).astype(np.uint8)


    _, binary_niblack = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow("image1",binary_niblack)
    cv2.waitKey(0)
            #imagex, contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(binary_niblack.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # modified by bigz
    # print(contours)
    # for countour in contours:
    #
    #     for point_ in countour:
    #         cv2.circle(img, (point_[0][0], point_[0][1]), 1, (255, 0, 0), 2)
    # cv2.imshow("image",img)
    # cv2.waitKey(0)

    dist_thres = 9999
    area_bounding_rect = 0
    for contour in contours:
                #用一个最小的矩形,把找到的形状包起来
        bdbox = cv2.boundingRect(contour)
        # if (bdbox[3]/float(bdbox[2])>0.7 and bdbox[3]*bdbox[2]>100 and bdbox[3]*bdbox[2]<1200) or (bdbox[3]/float(bdbox[2])>3 and bdbox[3]*bdbox[2]<100):
            # cv2.rectangle(rgb,(bdbox[0],bdbox[1]),(bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]),(255,0,0),1)
        img_draw = img.copy()
        area_ = bdbox[2] * bdbox[3]
        if area_ > area_bounding_rect:
            area_bounding_rect = area_

            points = np.array(contour[:, 0])

            # cv2.rectangle(img_draw, (bdbox[0],bdbox[1]), (bdbox[0]+bdbox[2],bdbox[1]+bdbox[3]), (255,255,255), 2)
            # cv2.imshow("image",img_draw)
            # cv2.waitKey(0)

    for point_ in points:
        cv2.circle(img, (point_[0], point_[1]), 1, (255, 0, 0), 2)
    cv2.imshow("image",img)
    cv2.waitKey(0)

    # #形状及大小筛选校验
    # det_x_max = 0
    # det_y_max = 0
    # num = 0
    # for i in range(len(contours)):
    #     x_min = np.min(contours[i][ :, :, 0])
    #     x_max = np.max(contours[i][ :, :, 0])
    #     y_min = np.min(contours[i][ :, :, 1])
    #     y_max = np.max(contours[i][ :, :, 1])
    #     det_x = x_max - x_min + 1
    #     det_y = y_max - y_min + 1
    #     print(x_min, x_max, y_min, y_max)
    #     if (det_x / det_y > 1.8) and (det_x > det_x_max ) and (det_y > det_y_max ):
    #         det_y_max = det_y
    #         det_x_max = det_x
    #         num = i
    # # 获取最可疑区域轮廓点集
    # points = np.array(contours[num][:, 0])

    # print(points)




    # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
    rect = cv2.minAreaRect(points)
    # 获取矩形四个顶点，浮点型
    box = cv2.boxPoints(rect)
    # 取整
    box = np.int0(box)

    # 获取四个顶点坐标
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    # 上下左右四个点坐标
    vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y], [right_point_x, right_point_y]])



    # 畸变情况1
    if rect[2] > -45:
        new_right_point_x = vertices[0, 0]
        new_right_point_y = int(vertices[1, 1] - (vertices[0, 0]- vertices[1, 0]) / (vertices[3, 0] - vertices[1, 0]+0.1) * (vertices[1, 1] - vertices[3, 1]))
        new_left_point_x = vertices[1, 0]
        new_left_point_y = int(vertices[0, 1] + (vertices[0, 0] - vertices[1, 0]) / (vertices[0, 0] - vertices[2, 0]+0.1) * (vertices[2, 1] - vertices[0, 1]))
        # 校正后的四个顶点坐标
        point_set_1 = np.float32([[440, 0],[0, 0],[0, 140],[440, 140]])
    # 畸变情况2
    elif rect[2] < -45:
        new_right_point_x = vertices[1, 0]
        new_right_point_y = int(vertices[0, 1] + (vertices[1, 0] - vertices[0, 0]) / (vertices[3, 0] - vertices[0, 0]+0.1) * (vertices[3, 1] - vertices[0, 1]))
        new_left_point_x = vertices[0, 0]
        new_left_point_y = int(vertices[1, 1] - (vertices[1, 0] - vertices[0, 0]) / (vertices[1, 0] - vertices[2, 0]+0.1) * (vertices[1, 1] - vertices[2, 1]))
        # 校正后的四个顶点坐标
        point_set_1 = np.float32([[0, 0],[0, 140],[440, 140],[440, 0]])

    # 校正前平行四边形四个顶点坐标
    new_box = np.array([(vertices[0, 0], vertices[0, 1]), (new_left_point_x, new_left_point_y), (vertices[1, 0], vertices[1, 1]), (new_right_point_x, new_right_point_y)])
    point_set_0 = np.float32(new_box)

    # 变换矩阵
    mat = cv2.getPerspectiveTransform(point_set_0, point_set_1)
    # 投影变换
    lic = cv2.warpPerspective(img, mat, (440, 140))
    print('plate correlation time',time.time()-tic)
    cv2.imshow('lic', lic)
    cv2.waitKey(0)
    return lic


