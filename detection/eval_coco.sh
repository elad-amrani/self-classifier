#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=sc_800ep_coco_eval
#SBATCH --time=0:20:00
#SBATCH --requeue
#SBATCH --mem=64G

master_node=${SLURM_NODELIST:0:3}${SLURM_NODELIST:4:3}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000

EXPERIMENT_PATH=${HOME}"/scratch/sc_experiments/sc_800ep_coco_eval"
MODEL_WEIGHTS=${HOME}"/scratch/sc_experiments/sc_800ep_coco_train/model_final.pth"
mkdir -p $EXPERIMENT_PATH

srun --output=${EXPERIMENT_PATH}/%j.out --error=${EXPERIMENT_PATH}/%j.err --label python -u ./train_net.py \
--config-file ./configs/coco_R_50_C4_2x_ssc.yaml \
--dist-url ${dist_url} \
--eval-only \
MODEL.WEIGHTS ${MODEL_WEIGHTS} \
OUTPUT_DIR ${EXPERIMENT_PATH}