#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=80G
#SBATCH --time=5-0:00:00 
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --output=moda.stdout
#SBATCH --job-name=moda


module restore deeplearning
source activate crossmoda2023
cd /nobackup/user/liuh26/crossmoda/src


A=(Basic Slerp Basic Slerp)
B=(v9 v9 v5 v5)


python -W ignore train.py -n ${A[$SLURM_ARRAY_TASK_ID]}_${B[$SLURM_ARRAY_TASK_ID]} -t ${A[$SLURM_ARRAY_TASK_ID]} -c 4 --max_epoch 2000 --skip_val_epoch 2000 --html_iter 50 --trans_ckpt ./models/pretrained_${B[$SLURM_ARRAY_TASK_ID]}.pth
