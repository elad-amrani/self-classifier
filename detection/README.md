
## Self-Classifier: Transferring to Detection

*Code was taken from [MoCo](https://github.com/facebookresearch/moco/tree/master/detection) repository.*

The `train_net.py` script reproduces the object detection experiments on Pascal VOC and COCO.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

1. Convert a pre-trained Self-Classifier model to detectron2's format:
   ```
   python convert-pretrain-to-detectron2.py input.pth.tar output.pkl
   ```

1. Put dataset under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

1. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_ssc.yaml \
	--num-gpus 8 MODEL.WEIGHTS ./output.pkl
   ```

### Results

#### COCO detection:

| pretrain | AP | AP50 | AP75 |
|----------|----|------|------|
|ImageNet-1M, supervised|38.2 | 58.2 | 41.2 |
|ImageNet-1M, Self-Classifier|41.5 | 61.3 | 45.0 |

#### COCO segmentation:

| pretrain | AP | AP50 | AP75 |
|----------|----|------|------|
|ImageNet-1M, supervised|33.3 | 54.7 | 35.2|
|ImageNet-1M, Self-Classifier|36.1 | 58.1 | 38.7 |

#### Pascal VOC 2007 test, fine-tuned on 2007+2012 trainval:

| pretrain | AP | AP50 | AP75 |
|----------|----|------|------|
|ImageNet-1M, supervised|53.5 | 81.3 | 58.8 |
|ImageNet-1M, Self-Classifier|56.6 | 82.4 | 62.6 |



