# C-volution: A Hybrid operator for Visual Recognition
### Accepted by PRCV2023

**The official repository with Pytorch**

We are working on the final version paper, and the code will be available soon!

## Abstract
Convolution is a fundamental building block of modern neural networks, playing a critical role in the success of deep learning for vision tasks. However, convolutional neural networks exhibit limited spatial context due to their local receptive field, which also neglects global/long-term dependent relations. To this end, we propose a lightweight hybrid structure operator, called C-volution. The operator utilizes a multi-branch architecture to extract spatial and channel information from input data separately, enabling the network to capture abstract features while preserving important spatial information. In addition, summarizes context information in a larger spatial range by generating dynamic kernels to strengthen the spatial contextual aggregation capability, overcoming the difficulty of long-term interactions in convolutions. This paper validates the efficacy of our operator through extensive experiments on ImageNet classification, COCO detection and segmentation, and the results have demonstrated the proposed C-volution when paired with ResNet50 achieves an outstanding boost in performance on visual tasks(+2.0\% top-1 accuracy, +3.1\% box mAP, and +2.0\% mask mAP) while having low parameters (i.e., CedNet50@16.3M Params).

## Getting Started
This repository is fully built upon the [OpenMMLab](https://openmmlab.com/) toolkits. For each individual task, the config and model files follow the same directory organization as [mmcls](https://github.com/open-mmlab/mmclassification), [mmdet](https://github.com/open-mmlab/mmdetection)respectively, so just copy-and-paste them to the corresponding locations to get started.

For example, in terms of evaluating detectors
```shell
git clone https://github.com/open-mmlab/mmdetection # and install

# copy model files
cp det/mmdet/models/backbones/* mmdetection/mmdet/models/backbones
cp det/mmdet/models/necks/* mmdetection/mmdet/models/necks
cp det/mmdet/models/dense_heads/* mmdetection/mmdet/models/dense_heads
cp det/mmdet/models/roi_heads/* mmdetection/mmdet/models/roi_heads
cp det/mmdet/models/roi_heads/mask_heads/* mmdetection/mmdet/models/roi_heads/mask_heads
cp det/mmdet/models/utils/* mmdetection/mmdet/models/utils
cp det/mmdet/datasets/* mmdetection/mmdet/datasets

# copy config files
cp det/configs/_base_/models/* mmdetection/configs/_base_/models
cp det/configs/_base_/schedules/* mmdetection/configs/_base_/schedules
cp det/configs/Cvolution mmdetection/configs -r

# evaluate checkpoints
cd mmdetection
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```
## Model Zoo

The parameters/FLOPs&#8595; and performance&#8593; compared to the convolution baselines are marked in the parentheses. Part of these checkpoints are obtained in our reimplementation runs, whose performance may show slight differences with those reported in our paper. Models are trained with 64 GPUs on ImageNet, 8 GPUs on COCO, and 4 GPUs on Cityscapes.

### Image Classification on ImageNet

|         Model         | Params(M) | FLOPs(G) | Top-1 (%) |  | Config | Download |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|:---------:|:--------:|

| RedNet-38             | 12.39<sub>(36.7%&#8595;)</sub>     | 2.22<sub>(31.3%&#8595;)</sub>     | 77.48 | 93.57 | [config](https://github.com/d-li14/involution/blob/main/cls/configs/rednet/rednet38_b32x64_warmup_coslr_imagenet.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ETZIquU7P3lDvru0OAPiTYIBAt-B__2LpP_NeB4sR0hJsg?e=b9Rbl0) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/Ed62YcJgC-NCp72NpEsMLGABkb7f-EkCQ1X-RyLmAMYoUQ?e=Hqetbj) |
| RedNet-50             | 15.54<sub>(39.5%&#8595;)</sub>     | 2.71<sub>(34.1%&#8595;)</sub>     | 78.35 | 94.13 | [config](https://github.com/d-li14/involution/blob/main/cls/configs/rednet/rednet50_b32x64_warmup_coslr_imagenet.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EZjRG3qUMu5IuR7YH4Giyc8B6koPvu6s8rOlIG8-BuFevg?e=f4ce5G) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ETL5NxDwnQpCldbJb906aOABjjuhZSquxKzK5xYQm-6Bhw?e=lOzEEf) |
| CedNet-38            | 12.40| 2.2   | 78.0 |   | [config](https://github.com/d-li14/involution/blob/main/cls/configs/rednet/rednet101_b32x64_warmup_coslr_imagenet.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EXAuVXdXz1xAg5eG-dkvwTUBkds2IOK1kglHtkMeGz5z_A?e=vHvh5y) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EbbiBxdZoZJFmTPSg9hW3BIBLRmRpfPa70nu8pi_8ddOSw?e=CdAV86) |
| CedNet-50           | 16.3  | 2.9  | 78.83 |  | [config](https://github.com/d-li14/involution/blob/main/cls/configs/rednet/rednet152_b32x64_warmup_coslr_imagenet.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ERxcS4wXUCtPl4uUnPoT9vcByzhLA0eHgDE-fw_EESfP0w?e=x0dZWB) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EYr2Yx-p4w1AuT-Q3E7M2m0BFhAGDoYvxps09vYy4Cnj3A?e=XGxzPF) |

Before finetuning on the following downstream tasks, download the ImageNet pre-trained [CedNet-50 weights](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EaVInpb6TGJApN6QCAWwKJAB3cK9Iz55QfJgmhhaV7yuHw?e=yuWxyI) and set the `pretrained` argument in `det/configs/_base_/models/*.py` or `seg/configs/_base_/models/*.py` to your local path.

### Object Detection and Instance Segmentation on COCO

#### Faster R-CNN
|    Backbone     |     Neck    |     Head    |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP | Config | Download |
| :-------------: | :---------: | :---------: | :-----: | :-----: |:---------:|:--------:| :----: | :------: | :--------: |
|    RedNet-50-FPN     | convolution | convolution | pytorch |   1x    | 31.6<sub>(23.9%&#8595;)</sub> | 177.9<sub>(14.1%&#8595;)</sub> | 39.5<sub>(1.8&#8593;)</sub>   | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/faster_rcnn_red50_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ESOJAF74jK5HrevtBdMDku0Bgf71nC7F4UcMmGWER5z1_w?e=qGPdA5) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ESYSpzei_INMn1wu5qa0Su8B9YxXf_rOtib5xHjb1y2alA?e=Qn3lyd) |
|    RedNet-50-FPN     |  involution | convolution | pytorch |   1x    | 29.5<sub>(28.9%&#8595;)</sub> | 135.0<sub>(34.8%&#8595;)</sub> | 40.2<sub>(2.5&#8593;)</sub>   | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/faster_rcnn_red50_neck_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EV90stAJIXxEnDRe0QM0lvwB_jm9jwqwHoBOVVOqosPHJw?e=0QoikN) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/Ec8z-SZbJTxJrAJ3FLq0PSsB1Q7T1dXLvhfHmegQqH7rqA?e=5O9jDY) |
|    CedNet-50-FPN     |  C-volution |  convolution | pytorch |   1x    | 29.8  | 92.3 | 40.5 | [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/faster_rcnn_red50_neck_fpn_head_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EeTwxsehR5VLhvf5TbTr8WwBmiNUwUeuXtbdOJlg0mFkmw?e=DL3gWX) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EUBsDdHQ10BKp8wW2aj2GHYBzhHtmW2BP65PIhn3KcSYqA?e=6dmNn7) |

#### Mask R-CNN
|    Backbone     |     Neck    |     Head    |  Style  | Lr schd | Params(M) | FLOPs(G) | box AP | mask AP | Config | Download |
| :-------------: | :---------: | :---------: | :-----: | :-----: |:---------:|:--------:| :----: | :-----: | :------: | :--------: |
|    RedNet-50-FPN     | convolution | convolution | pytorch |   1x    | 34.2<sub>(22.6%&#8595;)</sub> | 224.2<sub>(11.5%&#8595;)</sub> | 39.9<sub>(1.5&#8593;)</sub>   | 35.7<sub>(0.6&#8593;)</sub>    |  [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/mask_rcnn_red50_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EdheYm71X2pFu427_557zqcBmuKaLKEoU5R0Z2Kwo2alvg?e=qXShyW) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EQK-5qH_XxhHn4QnxmQbJ4cBL3sz9HqjS0EoybT2s1751g?e=4gpwK2) |
|    RedNet-50-FPN     |  involution | convolution | pytorch |   1x    | 32.2<sub>(27.1%&#8595;)</sub> | 181.3<sub>(28.5%&#8595;)</sub> | 40.8<sub>(2.4&#8593;)</sub>   | 36.4<sub>(1.3&#8593;)</sub>    |  [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/mask_rcnn_red50_neck_fpn_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EYYgUzXjJ3VBrscng-5QW_oB9wFK-dcqSDYB-LUXldFweg?e=idFEgd) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/ETWdfYuhjY5AlGkUH11rLl4BLk9zsyKgwAbay47TYzIU-w?e=6ey6cD) |
|    CedNet-50-FPN     |  C-volution |  convolution | pytorch |   1x    | 33.4 | 190.6 | 41.0  | 36.7  |  [config](https://github.com/d-li14/involution/blob/main/det/configs/involution/mask_rcnn_red50_neck_fpn_head_1x_coco.py) | [model](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/EZwtdWXX8sBLp7L__TrmkykBPEe7kJInbkbUblP3PxuURQ?e=09l25P) &#124; [log](https://hkustconnect-my.sharepoint.com/:u:/g/personal/dlibh_connect_ust_hk/Ebevxbj_0OtNkb3uCdpM0aoBeMQUABiQ0bDfZ9P9Jw1AZA?e=ZUcbUo) |


For more detailed guidance, please refer to the original [mmcls](https://github.com/open-mmlab/mmclassification), [mmdet](https://github.com/open-mmlab/mmdetection)tutorials.

Before finetuning on the following downstream tasks, download the ImageNet pre-trained [CedNet-50 weights](https://pan.baidu.com/s/1cV3PKT0eC-CYoojkC0ksSQ?) and set the `pretrained` argument to your local path.

Currently, we provide an memory-efficient implementation of the C-voluton operator based on [CuPy](https://cupy.dev/). Please install this library in advance. A customized CUDA kernel would bring about further acceleration on the hardware. Any contribution from the community regarding this is welcomed!

