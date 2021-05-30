#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sc_800ep_20_nn_eval
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --mem=128G

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/experiments/sc_800ep_20_nn_eval"
PRETRAINED_PATH=${HOME}"/scratch/experiments/sc_800ep_train/model_800.pth.tar"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/knn_eval.py \
-j 32 \
-b 512 \
--kk 20 \
--subset-size -1 \
--print-freq 16 \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}