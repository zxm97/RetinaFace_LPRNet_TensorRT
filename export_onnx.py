import numpy as np
import cv2
import torch
from model import build_model
from utils import cfg, load_config

config = './config/lpr.yml'
model_path = './workspace/lpr_20231102_all_types.ckpt'
ONNX_FILE_PATH = './workspace/lpr_20231102_all_types.onnx'
load_config(cfg, config)

model_fp32 = build_model(cfg)

def preprocess(img):
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1)
    img = cv2.resize(img, tuple(cfg.input_size))
    # cv2.imshow('plate', img)
    # cv2.waitKey(0)
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
    return img

ckpt = torch.load(model_path)
model_fp32.load_state_dict({key.replace('model.', ''): value for key, value in ckpt["state_dict"].items()})
model_fp32.eval()
model_fp32.cuda()

input = preprocess(r"E:/plate_recognition/CCPD2019/crop/val/20231024-blue-川A25T9P-1698159160454714.jpg").cuda()
output = model_fp32(input)

torch.onnx.export(model_fp32, input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True, opset_version=11,
                  )

# #加载onnx模型
# import onnxruntime as ort
# lprnet_onnx = ort.InferenceSession(ONNX_FILE_PATH)
#
# # pytorch推理
# dummy_torch = torch.randn(1,3,48,96).cuda()
# model_fp32.eval()
# with torch.no_grad():
#     torch_res = model_fp32(dummy_torch).cpu().numpy()
# #onnx推理
# dummy_np = dummy_torch.cpu().data.numpy()
# onnx_res = lprnet_onnx.run(["output"],{"input":dummy_np})[0];   #这个“127”是onnx导出后输出层名称，也有可能不同
#
# #比较结果
# try:
#     np.testing.assert_almost_equal(torch_res, onnx_res, decimal=4)
# except AssertionError:
#     print("The torch and onnx results are not equal at decimal=4")
# else:
#     print("The torch and onnx results are equal at decimal=4")
