#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sc_800ep_tsne_imagenet
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --mem=64G

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/experiments/sc_800ep_tsne_imagenet"
PRETRAINED_PATH=${HOME}"/scratch/experiments/sc_800ep_train/model_800.pth.tar"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/tsne_imagenet.py \
--num-plots 5 \
--num-samples -1 \
--num-classes 10 \
-j 128 \
-b 1024 \
--print-freq 16 \
--cls-size 1000 \
--num-cls 30 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--tau 0.1 \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}