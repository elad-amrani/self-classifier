#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sc_800ep_knn_eval
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --mem=128G

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/sc_experiments/sc_800ep_knn_eval"
PRETRAINED_PATH=${HOME}"/scratch/sc_experiments/sc_800ep_train/model_800.pth.tar"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/knn_eval.py \
-j 32 \
-b 512 \
--val-batch-size 16 \
--num-classes 1000 \
--tau 0.1 \
--cls-size 1000 2000 4000 8000 \
--num-cls 4 \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--use-bn \
--use-half \
--print-freq 16 \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}