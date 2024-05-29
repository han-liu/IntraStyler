#!/bin/bash
#SBATCH --mail-user=han.liu@vanderbilt.edu
#SBATCH --mail-type=FAIL
#SBATCH --account=vise_acc
#SBATCH --partition=turing
#SBATCH --mem=64G
#SBATCH --time=2-0:00:00 
#SBATCH --gres=gpu:1
#SBATCH --output=basic.stdout
#SBATCH --job-name=basic


module restore deeplearning
source activate crossmoda2023
cd /nobackup/user/liuh26/crossmoda/src

python -W ignore train.py -n base -c 4 --max_epoch 2000 --skip_val_epoch 2000 --html_iter 50