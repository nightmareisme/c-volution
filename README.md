# c-volution 
a lightweight mixed-branch operator to enhance neural networks' channel aggregation capability and spatial dynamic convolution kernel generation efficiency

Convolution is a primitive operator in deep neural networks, and its unique channel specificity promotes the flourishing development of computer vision. The proposal of involution has highlighted the advantages of spatial dimensions. However, information in both channel and spatial dimensions is crucial, and the lack of a computing module that shares parameters in both spatial and channel dimensions simultaneously remains a challenging problem in the field. To this end, we propose a lightweight hybrid structure operator, called C-volution. This approach strengthens the aggregation of channel information through a channel interaction branch and then utilizes this information to promote the generation of dynamic convolution kernels, thereby enhancing the learning of spatial context, and endowing convolution with more comprehensivechannel and spatial characteristics. Extensive experiments have shown that the proposed C-volution has superior performance and achieves an outstanding boost in performance on visual tasks (+1.8% top-1 accuracy on ImageNet classification, +3.1% box AP, and +2.0% mask AP on COCO detection and segmentation) while having low parameters (i.e., CedNet50@16.3M Params).

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
cp det/configs/involution mmdetection/configs -r

# evaluate checkpoints
cd mmdetection
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

For more detailed guidance, please refer to the original [mmcls](https://github.com/open-mmlab/mmclassification), [mmdet](https://github.com/open-mmlab/mmdetection)tutorials.

Before finetuning on the following downstream tasks, download the ImageNet pre-trained [CedNet-50 weights](https://pan.baidu.com/s/1cV3PKT0eC-CYoojkC0ksSQ?) and set the `pretrained` argument to your local path.

Currently, we provide an memory-efficient implementation of the C-voluton operator based on [CuPy](https://cupy.dev/). Please install this library in advance. A customized CUDA kernel would bring about further acceleration on the hardware. Any contribution from the community regarding this is welcomed!

