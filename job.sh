#!/bin/bash
#SBATCH --job-name=mlmal
#SBATCH --gres=gpu:48gb:1             # Number of GPUs (per node)
#SBATCH --mem=100G               # memory (per node)
#SBATCH --time=6-5:50            # time (DD-HH:MM)
#SBATCH --error=/home/mila/c/chris.emezue/MLM_AL/slurmerror-%j.txt
#SBATCH --output=/home/mila/c/chris.emezue/MLM_AL/slurmoutput-%j.txt

###########cluster information above this line


###load environment 


module load python/3
cd /home/mila/c/chris.emezue/MLM_AL
module load python/3.7 cuda/10.2/cudnn/7.6

source /home/mila/c/chris.emezue/scratch/mlmal/env/bin/activate
python active_learning.py \
    --experimet_path /home/mila/c/chris.emezue/scratch/mlmal/experiments_500ks \
    --data_folder /home/mila/c/chris.emezue/scratch/mlmal/data \
    active_learning_steps 5

# One liner with pdb
#python -m pdb active_learning.py --experiment_path /home/mila/c/chris.emezue/scratch/mlmal/experiments_500ks --data_folder /home/mila/c/chris.emezue/scratch/mlmal/data

