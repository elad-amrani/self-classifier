#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=obow_200ep_kmeans_eval
#SBATCH --time=0:10:00
#SBATCH --requeue
#SBATCH --mem=128G

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/sc_experiments/obow_200ep_kmeans_eval"
PRETRAINED_PATH="./pretrained/obow_200ep.pth.tar"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/kmeans_eval.py \
--model obow \
-j 32 \
-b 128 \
--num-classes 1000 \
--kk 1000 \
--print-freq 16 \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
${DATASET_PATH}