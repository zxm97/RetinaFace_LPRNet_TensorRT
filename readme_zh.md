
**[English](readme.md), [中文](readme_zh.md)**

## 支持的车牌种类

### 检测

目前训练数据大部分是蓝牌和单层绿牌，加上少量单层黄牌、双层黄牌、单层白牌，对蓝牌和单层绿牌之外的车牌目标检测和关键点检测不一定准确。

### 识别

训练数据包括 CCPD2019、CCPD2020、CBLPRD-330k 和用代码合成的车牌 (合成参考<https://github.com/ChHanXiao/license-plate-recoginition>)。

种类包含蓝色单层车牌，黄色单层车牌，绿色新能源车牌，民航车牌，黑色单层车牌，白色警牌、军牌、武警车牌，
黄色双层车牌，绿色农用车牌，白色双层军牌。

## 环境
### PC
Windows 10

python 3.7.10

CUDA 11.7

cuDNN 8.6.0.163

TensorRT 8.5.1.7

OpenCV 4.6.0 (build from source)

CMake 3.26.5

Visual Studio 16 2019


### Jetson Nano
python 3.6.9

Jetpack 4.6.2

CUDA 10.2.89

cuDNN 8.2.1.32

TensorRT 8.2.1.8

OpenCV 4.5.2 with GStreamer support

## Requirements
### PC
albumentations==1.3.1

imutils==0.5.4

matplotlib==3.3.4

numpy==1.21.6

opencv_python==4.8.1.78

opencv_python_headless==4.8.1.78

Pillow==9.5.0

pycuda==2021.1+cuda115

pytorch_lightning==1.8.0

PyYAML==5.4.1

task==0.2.5

termcolor==2.3.0

torch==1.13.1+cu116

torchvision==0.14.1+cu116

### Jetson Nano
albumentations==1.3.1

imutils==0.5.4

matplotlib==3.3.2

numpy==1.16.1

Pillow==10.1.0

pycuda==2021.1

pytorch_lightning==1.5.0

PyYAML==6.0.1

task==0.2.5

termcolor==1.1.0

torch==1.6.0a0+b31f58d

torchvision==0.9.0a0+01dfa8e




## 使用方法
### 在PC上运行 PyTorch demo:
运行 demo_torch.py

### 在PC上运行 TensorRT demo (Python):
#### 第 1 步
克隆以下项目，根据说明构建 Retinaface trt engine:

https://github.com/zxm97/Pytorch_Retina_License_Plate_trt
#### 第 2 步
克隆以下项目，根据说明构建 LPRNet trt engine:

https://github.com/zxm97/license-plate-recoginition_trt

#### 第 3 步
运行 demo_trt.py 或 demo_trt_fpn_reduced.py 或 demo_trt_fpn_reduced_async.py

 - demo_trt.py 是未修改的RetinaFace模型
 - demo_trt_fpn_reduced.py 中的 RetinaFace 模型去掉了最大的一层特征图，FPN只融合两层，减少了锚框数量
 - demo_trt_fpn_reduced_async.py 在 demo_trt_fpn_reduced.py 的基础上，把图片读取或视频解码和检测的预处理部分放在一个线程，把检测模型的推理、检测模型后处理、车牌裁剪和矫正、识别模型的预处理、识别模型的推理、识别模型的后处理放在另一个线程，提高吞吐量。
  
### 在PC上运行 TensorRT demo (C++):

#### 第 1 步
克隆以下项目并运行 gen_wts_for_tensorrtx.py 来生成 weight map 文件:

https://github.com/zxm97/Pytorch_Retina_License_Plate_trt
#### 第 2 步
打开 CMake (GUI)

把 source code directory 设为 xxx/Project_cpp

把 binaries directory 设为 xxx/Project_cpp/build

依次点击 Configure、 Generate、 Open Project，在 Visual Studio 中打开

#### 第 3 步
 选择 Release x64, 生成 decodeplugin, 成功后会出现 xxx\Project_cpp\build\Release\decodeplugin.lib (注意生成 .lib 而不是 .dll)

#### 第 4 步
生成 retina_mnet_plate

把第 1 步的 weight map 文件 移到 xxx\Project_cpp\build\Release

打开 Windows PowerShell, 执行以下命令来构建 RetinaFace trt engine:

`./retina_mnet_plate.exe -s`

#### 第 5 步
克隆以下项目，根据说明构建 LPRNet trt engine:

https://github.com/zxm97/license-plate-recoginition_trt

把 LPRNet trt engine 文件移到 xxx\Project_cpp\build\Release

#### 第 6 步
生成 demo

执行以下命令，在图片上进行车牌检测和识别:

`./demo.exe -d`


![img](result_cpp.jpg)

### 在 Jetson Nano 上运行TensorRT demo (Python):


#### 第 1 步
克隆以下项目，根据说明构建 Retinaface trt engine (用去掉了最大的一层特征图的模型权重):

https://github.com/zxm97/Pytorch_Retina_License_Plate_trt
#### 第 2 步
克隆以下项目，根据说明构建 LPRNet trt engine:

https://github.com/zxm97/license-plate-recoginition_trt

#### 第 3 步
运行 demo_trt_fpn_reduced_async.py

 - 把图片读取或视频解码和检测的预处理部分放在一个线程，把检测模型的推理、检测模型后处理、车牌裁剪和矫正、识别模型的预处理、识别模型的推理、识别模型的后处理放在另一个线程，提高吞吐量
 - 用 Jetson Nano 的硬解码器来解码视频，降低 cpu 使用率 (Jetson Nano 的 A57 性能太弱)
 - 在自己截取的 1258 x 684 h.264 视频上测试，整个系统能达到每秒处理20帧


## 参考

https://github.com/gm19900510/Pytorch_Retina_License_Plate

https://github.com/ChHanXiao/license-plate-recoginition

https://github.com/1996scarlet/faster-mobile-retinaface

https://github.com/wang-xinyu/tensorrtx

https://github.com/SunlifeV/CBLPRD-330k

https://github.com/yxgong0/CRPD