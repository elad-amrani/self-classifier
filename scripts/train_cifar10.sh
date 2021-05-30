#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sc_800ep_train_cifar10
#SBATCH --time=06:00:00
#SBATCH --mem=32G

EXPERIMENT_PATH=${HOME}"/scratch/experiments_cifar/sc_800ep_train_cifar10"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/train_cifar.py \
-j 8 \
-b 256 \
--print-freq 16 \
--epochs 800 \
--lr 3.2 \
--final-lr 0.032 \
--wd 1e-6 \
--cls-size 30 \
--num-cls 30 \
--dim 64 \
--hidden-dim 256 \
--num-hidden 2 \
--tau 0.3 \
--sgd \
--cos \
--save-path ${EXPERIMENT_PATH} \
--data CIFAR10