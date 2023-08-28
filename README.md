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

For more detailed guidance, please refer to the original [mmcls](https://github.com/open-mmlab/mmclassification), [mmdet](https://github.com/open-mmlab/mmdetection)tutorials.

Before finetuning on the following downstream tasks, download the ImageNet pre-trained [CedNet-50 weights](https://pan.baidu.com/s/1cV3PKT0eC-CYoojkC0ksSQ?) and set the `pretrained` argument to your local path.

Currently, we provide an memory-efficient implementation of the C-voluton operator based on [CuPy](https://cupy.dev/). Please install this library in advance. A customized CUDA kernel would bring about further acceleration on the hardware. Any contribution from the community regarding this is welcomed!

