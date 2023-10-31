import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward_slow(self):
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
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


    def forward(self):
        # 向量化的版本
        anchors = []
        print('self.feature_maps:', self.feature_maps) # [[80, 148], [40, 74], [20, 37]]
        for k, f in enumerate(self.feature_maps):
            min_size = self.min_sizes[k] # 'min_sizes': [[16, 32], [64, 128], [256, 512]],
            mat = np.array(list(product(range(f[0]), range(f[1]), min_size))).astype(np.float32) # product(A, B) 和 ((x,y) for x in A for y in B)一样.
            mat[:, 0], mat[:, 1] = ((mat[:, 1] + 0.5) * self.steps[k] / self.image_size[1],
                                    (mat[:, 0] + 0.5) * self.steps[k] / self.image_size[0])
            mat = np.concatenate([mat, mat[:, 2:3]], axis=1)
            mat[:, 2] = mat[:, 2] / self.image_size[1]
            mat[:, 3] = mat[:, 3] / self.image_size[0]
            anchors.append(mat)
        print('anchors', len(anchors), anchors[0].shape)
        output = np.concatenate(anchors, axis=0)
        if self.clip:
            output = np.clip(output, 0, 1)
        return torch.from_numpy(output)
