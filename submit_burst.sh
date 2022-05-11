#!/bin/bash

#SBATCH --time=00-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output="sbatch_logs/%A_%x.txt"
#SBATCH --account=csci-ga-2565-2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --job-name=finetune_rec


singularity exec \
    --nv --overlay /scratch/nv2099/overlay-50G-10M.ext3:ro \
    --bind /scratch/nv2099 --bind /scratch_tmp/nv2099 \
    /scratch/nv2099/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh;
    conda activate vit_mae;
    cd /home/nv2099/projects/vit_mae/mae;
    # python -u main_pretrain.py --loss_type both --batch_size 40 \
    #     --output_dir runs/scratch/${SLURM_JOB_NAME}_${SLURM_JOB_ID};
    python -u main_finetune.py --batch_size 20 --finetune runs/scratch/pretrain_rec_77226/checkpoint-49.pth \
        --output_dir runs/scratch/${SLURM_JOB_NAME}_${SLURM_JOB_ID};
    "


# Pretrain Script
# python main_pretrain.py --batch_size 40 --loss_type both
# Finetune Script
# python main_finetune.py --batch_size 40 --finetune runs/pretrain_cifar/both/checkpoint-49.pth