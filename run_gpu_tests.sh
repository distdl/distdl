#!/bin/bash

#SBATCH --nodes 4
#SBATCH --ntasks-per-node=5
#SBATCH --time 00:10:00
#SBATCH --partition t4_normal_q
#SBATCH --account distdl
#SBATCH --output=output.log
#SBATCH --gres=gpu:1

#Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Load modules silently
{
    module load mpi4py/3.0.2-gompi-2020a-timed-pingpong
    module load OpenMPI/4.0.5-gcccuda-2020b
    module load PyTorch/1.7.1-fosscuda-2020b
} &> /dev/null

mpirun --n 20 -x USE_CUDA=1 python -B -m mpi4py -m pytest --with-mpi -rsa -x \
    tests/test_transpose.py \
    tests/test_halo_exchange.py \
    tests/test_sum_reduce.py \
    tests/test_broadcast.py \
    tests/test_padnd.py \
    tests/test_conv_feature.py \
    tests/test_batchnorm.py
