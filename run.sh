#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --mem=4GB
#SBATCH --mail-type=END
#SBATCH --mail-user=cindy.bui@nyu.edu
##SBATCH --gres=gpu:p100:1
##SBATCH --gres=gpu:k80:1
##SBATCH --gres=gpu:p1080:1
##SBATCH --gres=gpu:p40:1
#SBATCH --gres=gpu:v100:1


# k80, p1080, p40, p100 and v100.
module load cuda/10.1.105
module load libsndfile/intel/1.0.28   
module load gcc/6.3.0
nvidia-smi
build/executable.out 18040
