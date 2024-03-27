#!/bin/bash -l                 
#
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
                                 
#unset SLURM_EXPORT_ENV             # enable export of environment from this script to srun
            
module load python/3.8-anaconda cuda/11.1.0 cudnn/8.0.5.39-cuda11.1 # Load necessary modules

cd ${HOME}/DeepSolo
conda activate deepsolo
python -c 'import torch; print(torch.rand(2,3).cuda())'
