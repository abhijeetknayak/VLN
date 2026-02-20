#!/bin/bash -x
#SBATCH --account=m3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --partition=booster
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4

set -euo pipefail
mkdir -p logs

# ===== USER CONFIG =====
CONTAINER=/e/scratch/m3/nav/vln_aarch64.sif
CONDA_ENV=vln3_12
FOLDER=/e/scratch/m3/nav/VLN
SCRIPT_IN_CONTAINER=scripts/train/qwenvl_train/with_poses/train_poses.sh     # path INSIDE container
BIND="--bind $PWD:$PWD"  # bind current directory (adjust as needed)
# =======================

# Slurm-derived settings
export NNODES="${SLURM_JOB_NUM_NODES}"

# Usually: number of GPUs you requested per node
# Keep this explicit unless your cluster reliably sets SLURM_GPUS_ON_NODE.
export NPROC_PER_NODE="4"

# Rendezvous endpoint: pick first node in the allocation as "master addr"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# A port that’s stable within the job; deriving from job id reduces collision risk
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

# MUST be same on all nodes; job id is perfect
export RDZV_ID="${SLURM_JOB_ID}"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "RDZV_ID=$RDZV_ID"
echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE"

############################## WANDB ###############################
# Write W&B files to job-local storage (change to your fast scratch)
export WANDB_DIR="$FOLDER/wandb"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"

# Offline mode: never tries to contact wandb servers
export WANDB_MODE=offline

# (Optional) avoid background service quirks on restricted systems
export WANDB_START_METHOD=thread
####################################################################

cd "$FOLDER"

# One process per node; each process starts torchrun which spawns NPROC_PER_NODE workers on that node.
srun --ntasks="$NNODES" --ntasks-per-node=1 \
  apptainer exec --nv $BIND "$CONTAINER" bash -lc "
    set -euo pipefail
    export NNODES='${NNODES}'
    export NPROC_PER_NODE='${NPROC_PER_NODE}'
    export MASTER_ADDR='${MASTER_ADDR}'
    export MASTER_PORT='${MASTER_PORT}'
    export RDZV_ID='${RDZV_ID}'

    export WANDB_DIR='$WANDB_DIR'
    export WANDB_CACHE_DIR='$WANDB_CACHE_DIR'
    export WANDB_CONFIG_DIR='$WANDB_CONFIG_DIR'
    export WANDB_MODE='$WANDB_MODE'
    export WANDB_START_METHOD='$WANDB_START_METHOD'
    
    source /root/env.sh 
    conda activate '$CONDA_ENV'
    cd "$FOLDER"
    bash '$SCRIPT_IN_CONTAINER'
  "
