# Self-Classifier: Self-Supervised Classification Network

Official PyTorch implementation and pretrained models of the paper [Self-Supervised Classification Network](https://arxiv.org/abs/2103.10994) from ECCV 2022.

<p align="center">
<img src="graphics/Self-Classifier_arch.jpg" width="65%">
</p>

**Self-Classifier architecture**. Two augmented views of the same image are processed by a shared network comprised of a backbone (e.g. CNN) and a classifier (e.g. projection MLP + linear classification head). The cross-entropy of the two views is minimized to promote same class prediction while avoiding degenerate solutions by asserting a uniform prior on class predictions. The resulting model learns representations and discovers the underlying classes in a single-stage end-to-end unsupervised manner.

If you find this repository useful in your research, please cite:

    @article{amrani2021self,
      title={Self-Supervised Classification Network},
      author={Amrani, Elad and Karlinsky, Leonid and Bronstein, Alex},
      journal={arXiv preprint arXiv:2103.10994},
      year={2021}
    }
    
## Pretrained Models

Download pretrained models [here](https://drive.google.com/drive/folders/1o8M1cSVnhsMsdiRejj0DweWs2mhtP57F?usp=sharing).

## Setup

1. Install Conda environment:

        conda env create -f ./environment.yml

2. Install Apex with CUDA extension:
 
        export TORCH_CUDA_ARCH_LIST="7.0"  # see https://en.wikipedia.org/wiki/CUDA#GPUs_supported
        pip install git+git://github.com/NVIDIA/apex.git@4a1aa97e31ca87514e17c3cd3bbc03f4204579d0 --install-option="--cuda_ext"         


## Training & Evaluation

Distributed training & evaluation is available via Slurm. See SBATCH scripts [here](./scripts). 

**IMPORTANT**: set DATASET_PATH, EXPERIMENT_PATH and PRETRAINED_PATH to match your local paths.
 

### Training

| method          | epochs | NMI | AMI | ARI | ACC | linear probing top-1 acc. | training script |
|-----------------|--------|-----|-----|-----|-----|---------------------------|-----------------|
| Self-Classifier | 100    | 71.2| 49.2| 26.1| 37.3|                      72.4 |[script](./scripts/train_100ep.sh)|
| Self-Classifier | 200    | 72.5| 51.6| 28.1| 39.4|                      73.5 |[script](./scripts/train_200ep.sh)|
| Self-Classifier | 400    | 72.9| 52.3| 28.8| 40.2|                      74.2 |[script](./scripts/train_400ep.sh)|
| Self-Classifier | 800    | 73.3| 53.1| 29.5| 41.1|                      74.1 |[script](./scripts/train_800ep.sh)|

NMI: Normalized Mutual Information, AMI: Adjusted Normalized Mutual Information, ARI: Adjusted Rand-Index and ACC: Unsupervised clustering accuracy. 
linear probing: training a supervised linear classifier on top of frozen self-supervised features.

### Evaluation

#### Unsupervised Image Classification

| dataset | NMI | AMI | ARI | ACC | evaluation script |
|---------|-----|-----|-----|-----|--------|
|ImageNet 1K classes| 73.3| 53.1| 29.5| 41.1 |[script](./scripts/unsupervised_cls_eval/cls_eval.sh)|
|ImageNet 10 superclasses (level #2 in hierarchy)| 74.0|  54.3| 30.9| 85.7|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_lvl_2.sh)|
|ImageNet 29 superclasses (level #3 in hierarchy)| 74.0|  54.3| 30.9| 79.7|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_lvl_3.sh)|
|ImageNet 128 superclasses (level #4 in hierarchy)| 74.0|  54.3| 30.9| 71.8|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_lvl_4.sh)|
|ImageNet 466 superclasses (level #5 in hierarchy)| 73.9|  54.3| 30.8| 60.0|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_lvl_5.sh)|
|ImageNet 591 superclasses (level #6 in hierarchy)|  74.1|  55.3| 32.1| 46.7|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_lvl_6.sh)|
|BREEDS Entity13 (ImageNet based)| 73.6| 54.1| 30.7| 84.4|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_entity13.sh)|
|BREEDS Entity30 (ImageNet based)| 72.9| 53.4| 29.8| 81.0|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_entity30.sh)|
|BREEDS Living17 (ImageNet based)| 67.2| 51.8| 26.4| 90.8|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_living17.sh)|
|BREEDS Nonliving26 (ImageNet based)| 72.2| 57.0| 36.8| 76.7|[script](./scripts/unsupervised_cls_eval/cls_eval_superclass_nonliving26.sh)|

NMI: Normalized Mutual Information, AMI: Adjusted Normalized Mutual Information, ARI: Adjusted Rand-Index and ACC: Unsupervised clustering accuracy.

#### K-Means Baselines Using Self-Supervised Pretrained Models

| method | NMI | AMI | ARI | ACC | evaluation script |
|---------|-----|-----|-----|-----|--------|
|BarlowTwins| 68.8 | 48.3 | 14.7 | 33.2|[script](./scripts/kmeans_baselines/kmneas_eval_barlowtwins.sh)|
|OBoW| 66.5 | 42.0 | 16.9 | 31.1|[script](./scripts/kmeans_baselines/kmneas_eval_obow.sh)|
|DINO| 66.2 | 42.3 | 15.6 | 30.7|[script](./scripts/kmeans_baselines/kmneas_eval_dino.sh)|
|MoCov2| 66.6 | 45.3 | 12.0 | 30.6|[script](./scripts/kmeans_baselines/kmneas_eval_mocov2.sh)|
|SwAV| 64.1 | 38.8 | 13.4 | 28.1|[script](./scripts/kmeans_baselines/kmneas_eval_swav.sh)|
|SimSiam| 62.2 | 34.9 | 11.6 | 24.9|[script](./scripts/kmeans_baselines/kmneas_eval_simsiam.sh)|

NMI: Normalized Mutual Information, AMI: Adjusted Normalized Mutual Information, ARI: Adjusted Rand-Index and ACC: Unsupervised clustering accuracy.
All methods are evaluated on ImageNet 1K classes with original pre-trained models - 
[MoCov2](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar), 
[OBoW](https://github.com/valeoai/obow/releases/download/v0.1.0/ImageNetFull_ResNet50_OBoW_full_feature_extractor.zip), 
[SimSiam](https://dl.fbaipublicfiles.com/simsiam/models/100ep-256bs/pretrain/checkpoint_0099.pth.tar), 
[SwAV](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar). 
DINO and BarlowTwins use PyTorch Hub (i.e., no need for direct download).

#### Image Classification with Linear Models

For training a supervised linear classifier on a frozen backbone, run:

        sbatch ./scripts/lincls_eval.sh

        
#### Image Classification with kNN

For running K-nearest neighbor classifier on ImageNet validation set, run:

        sbatch ./scripts/knn_eval.sh

#### Transferring to Object Detection and Instance Segmentation

See [./detection](./detection).


#### Ablation study

For training the 100-epoch ablation study baseline, run:

        sbatch ./scripts/ablation/train_100ep.sh

For training any of the ablation study runs presented in the paper, run:

        sbatch ./scripts/ablation/<ablation_name>/<ablation_script>.sh
        
## Qualitative Examples (classes predicted by Self-Classifier on ImageNet validation set)

<img src="graphics/grid_0.jpg" width="18%"> <img src="graphics/grid_1.jpg" width="18%"> <img src="graphics/grid_2.jpg" width="18%"> <img src="graphics/grid_3.jpg" width="18%"> <img src="graphics/grid_4.jpg" width="18%">
<img src="graphics/grid_5.jpg" width="18%"> <img src="graphics/grid_6.jpg" width="18%"> <img src="graphics/grid_7.jpg" width="18%"> <img src="graphics/grid_8.jpg" width="18%"> <img src="graphics/grid_9.jpg" width="18%">
<img src="graphics/grid_10.jpg" width="18%"> <img src="graphics/grid_11.jpg" width="18%"> <img src="graphics/grid_12.jpg" width="18%"> <img src="graphics/grid_13.jpg" width="18%"> <img src="graphics/grid_14.jpg" width="18%">

**High accuracy classes predicted by Self-Classifier on ImageNet validation set**. Images are sampled randomly from each predicted class. Note that the predicted classes capture a large variety of different backgrounds and viewpoints.

To reproduce qualitative examples, run:

        sbatch ./scripts/cls_eval.sh

## License

See the [LICENSE](./LICENSE) file for more details.
