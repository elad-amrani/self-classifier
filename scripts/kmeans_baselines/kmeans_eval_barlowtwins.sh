#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=barlowtwins_800ep_kmeans_eval
#SBATCH --time=0:10:00
#SBATCH --requeue
#SBATCH --mem=128G

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/sc_experiments/barlowtwins_800ep_kmeans_eval"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/kmeans_eval.py \
--model barlowtwins \
-j 32 \
-b 128 \
--num-classes 1000 \
--kk 1000 \
--print-freq 16 \
--save-path ${EXPERIMENT_PATH} \
${DATASET_PATH}