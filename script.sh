#!/bin/bash
#
#SBATCH --job-name=sbatch-tf-gpu-
#SBATCH --output=tf-train.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=70:00:00
#SBATCH --mail-user=*****school_email*****
#SBATCH --mail-type=END

module --ignore-cache load python3/3.6.6
module load cuda/10.0
nvidia-smi

python -m pip install --user virtualenv
python -m virtualenv pyenv
source pyenv/bin/activate

module load python3/3.6.6 > tf-train.log
module load cuda/10.0 >> tf-train.log
nvidia-smi >> tf-train.log

python -m pip install -r requirements.txt

python -c "import torch; print(torch.cuda.current_device())" >> tf-train.log
python -c "import torch; print(torch.cuda.device(0))" >> tf-train.log
python -c "import torch; print(torch.cuda.device_count())" >> tf-train.log
python -c "import torch; print(torch.cuda.get_device_name(0))" >> tf-train.log
python -c "import torch; print(torch.cuda.is_available())" >> tf-train.log
python -m pip list >> tf-train.log
nvidia-smi >> tf-train.log
nvcc --version >> tf-train.log


python /home/*****ID*****/distracting_features/distracting_feature/main.py --net 'Reab3p16' --datapath '/home/*****ID*****/distracting_features/HPC_Transfer/RAVEN-10000/' --rl True --typeloss True --model_dir '/home/*****ID*****/distracting_features/output' > '/home/*****ID*****/distracting_features/output/output.txt'

deactivate