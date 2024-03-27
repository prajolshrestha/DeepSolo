#!/bin/bash -l                 
#
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --time=06:00:00
#SBATCH --export=NONE
                                 
unset SLURM_EXPORT_ENV            
            
module load python/3.8-anaconda cuda/11.1.0 cudnn/8.0.5.39-cuda11.1 # Load necessary modules

cd ${HOME}/practice/DeepSolo
conda activate deepsolo_practice

python tools/train_net.py --config-file configs/ViTAEv2_S/Map/finetune_map.yaml --num-gpus 1