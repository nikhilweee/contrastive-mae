#!/bin/bash
# Add -e to stop on errors
# Add -x to print before executing

#SBATCH --mem=64GB
#SBATCH --time=00-12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --output="sbatch_logs/%A_%x.txt"
#SBATCH --job-name=pretrain_cifar_192_both

# The following options will not be applied
# SBATCH --nodelist="gr*"
# SBATCH --gres=gpu:rtx8000:1


singularity exec \
    --nv --overlay /scratch/nv2099/images/overlay-50G-10M.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh;
    conda activate vit_mae;
    cd /home/nv2099/projects/vit_mae/mae;
    python -u main_pretrain.py --epochs 50 --batch_size 192 --warmup_epochs 5 --loss_type both --output_dir runs/pretrain_cifar_both;
    "
