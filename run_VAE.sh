#!/bin/bash
#----------------------------------------------------------------
# running a multiple independent jobs
#----------------------------------------------------------------
#SBATCH --job-name=VAE
#SBATCH --output=logs/%A_%a_vae.log
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00
#SBATCH --mem=400G
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --account=kondrgrp
#SBATCH --partition=gpu
#SBATCH --exclude=gpu62
#SBATCH --constraint=GTX1080Ti 
#SBATCH --mail-user=emaksimo@ist.ac.at
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1

#print out the list of GPUs before the job is started
srun /usr/bin/nvidia-smi
# get comma-separated list of available devices
export CUDA_VISIBLE_DEVICES=$(srun nvidia-smi pmon -c 1 | awk '/^[^#]/ {if ($2=="-") {printf("%s,",$1);}}')
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES::-1} 
echo $CUDA_VISIBLE_DEVICES

# Arguments: batch_size, num_epochs,jobID, replicate_n
module load python/3.8
prog='run_VAE_model.py'

case $SLURM_ARRAY_TASK_ID in
    1)
    srun python $prog -b 10 -e 50 -j $SLURM_JOB_ID -r 0 --n_lr 1 --min_lr 17e-4 --max_lr 1e-1 --n_beta 10 --min_beta 1e-6 --max_beta 1
    ;;
    *)
    echo -n "Something went wrong with specifying bash cases"
    ;;
esac


