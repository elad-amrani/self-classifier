#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sc_800ep_coco_train
#SBATCH --time=48:00:00
#SBATCH --qos=dcs-48hr
#SBATCH --mem=64G

master_node=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:3}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

EXPERIMENT_PATH=${HOME}"/scratch/sc_experiments/sc_800ep_coco_train"
MODEL_WEIGHTS="./sc_800ep_train.pkl"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./train_net.py \
--config-file ./configs/coco_R_50_C4_2x_ssc.yaml \
--dist-url ${dist_url} \
--resume \
MODEL.WEIGHTS ${MODEL_WEIGHTS} \
OUTPUT_DIR ${EXPERIMENT_PATH}