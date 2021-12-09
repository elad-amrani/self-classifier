#!/usr/bin/env bash
#SBATCH --nodes=11
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=16
#SBATCH --job-name=sc_100ep_train_dim_256
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --mem=64G

master_node=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:3}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

DATASET_PATH=${HOME}"/scratch/imagenet/"
EXPERIMENT_PATH=${HOME}"/scratch/sc_experiments/sc_100ep_train_dim_256"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./src/train.py \
--syncbn_process_group_size 22 \
-j 32 \
-b 372 \
--print-freq 16 \
--epochs 100 \
--lr 4.8 \
--start-warmup 0.3 \
--final-lr 0.0048 \
--lars \
--sgd \
--cos \
--wd 1e-6 \
--cls-size 1000 2000 4000 8000 \
--num-cls 4 \
--queue-len 262144 \
--dim 256 \
--hidden-dim 4096 \
--num-hidden 2 \
--row-tau 0.1 \
--col-tau 0.05 \
--global-crops-scale 0.4 1.0 \
--local-crops-scale 0.05 0.4 \
--local-crops-number 6 \
--use-amp \
--save-path ${EXPERIMENT_PATH} \
--dist-url ${dist_url} \
${DATASET_PATH}