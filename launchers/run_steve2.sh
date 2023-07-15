#! /bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -J steve_res
#SBATCH --exclude=DemoRTX,ServerA100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH -o ./reports/steve_res%j.out

source $(conda info --base)/bin/activate
conda activate torch12_cuda102

python tools/run_net.py --cfg configs/movi_e/base_res_sl.yaml