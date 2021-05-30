#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sc_100ep_train_learnable_cls
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --mem=64G

master_node=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:3}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/experiments/sc_100ep_train_learnable_cls"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/train.py \
--syncbn_process_group_size 4 \
-j 32 \
-b 1024 \
--print-freq 16 \
--epochs 100 \
--lr 4.8 \
--start-warmup 0.3 \
--final-lr 0.0048 \
--lars \
--wd 1e-4 \
--cls-size 1000 \
--num-cls 30 \
--learnable-cls \
--dim 128 \
--hidden-dim 4096 \
--num-hidden 2 \
--tau 0.1 \
--sgd \
--cos \
--use-aug \
--use-amp \
--save-path ${EXPERIMENT_PATH} \
--dist-url ${dist_url} \
${DATASET_PATH}