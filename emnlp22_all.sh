#!/bin/bash
#SBATCH --job-name=emnlp_al_mlm
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --cpus-per-gpu=18
#SBATCH --mem=128G
#SBATCH --time=168:00:00         
#SBATCH --partition=long
#SBATCH --error=/home/mila/b/bonaventure.dossou/emnlp22/slurmerror.txt
#SBATCH --output=/home/mila/b/bonaventure.dossou/emnlp22/slurmoutput.txt


###########cluster information above this line
cd /home/mila/b/bonaventure.dossou/
module load python/3.7 cuda/10.2/cudnn/7.6 && virtualenv /home/mila/b/bonaventure.dossou/env && source /home/mila/b/bonaventure.dossou/env/bin/activate
cd emnlp22/
python active_learning.py