#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sc_800ep_lincls
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --mem=64G

master_node=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:3}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/experiments/sc_800ep_lincls"
PRETRAINED_PATH=${HOME}"/scratch/experiments/sc_800ep_train/model_800.pth.tar"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/eval.py \
-j 32 \
-b 1024 \
--print-freq 16 \
--epochs 100 \
--wd 0.0 \
--lr 0.8 \
--lars \
--sgd \
--cos \
--save-path ${EXPERIMENT_PATH} \
--pretrained ${PRETRAINED_PATH} \
--dist-url ${dist_url} \
${DATASET_PATH}