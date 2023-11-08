#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Defines a `load_data` function that returns a generator yielding
feed_dicts so that this script can be used as the argument for
the --data-loader-script command-line parameter.
"""
import os
import random
import numpy as np
import cv2
from utils import cfg, load_config

calib_num_images = 10000
config = './config/lpr.yml'
load_config(cfg, config)

dataset_path = r"E:/plate_recognition/CBLPRD-330k_v1/train"

imgs = os.listdir(dataset_path)
random.shuffle(imgs)

def load_data():
    for img_ind, img_name in enumerate(imgs):
        if img_ind > (calib_num_images - 1):
            break
        img = cv2.imdecode(np.fromfile(os.path.join(dataset_path, img_name), dtype=np.uint8), 1) # H, W, C
        img = cv2.resize(img, tuple(cfg.input_size))
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        print(img_ind+1, 'of', calib_num_images)
        yield {"input": img}  # Still totally real data

# if __name__ == "__main__":
#     loader = load_data()
#     print(next(loader))
#     print(next(loader))
#     print(next(loader))



