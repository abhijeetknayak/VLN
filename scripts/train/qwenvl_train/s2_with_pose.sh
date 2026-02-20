#!/usr/bin/env bash
set -euo pipefail


FOLDER=/e/scratch/m3/nav/VLN
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


# Distributed training configuration
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
RDZV_ID="${RDZV_ID:-12345}"
NNODES="${NNODES:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Restrict to exactly 4 local GPUs (optional but recommended)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero2.json

# Model configuration
llm=checkpoints/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=2e-5
vision_tower_lr=5e-6
batch_size=6
grad_accum_steps=1
max_pixels=313600
min_pixels=3136

# Dataset configuration (replace with public dataset names)
vln_datasets=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30,scalevln_125cm_0_30,scalevln_60cm_30_30
# vln_datasets=r2r_125cm_0_30

# Output configuration
run_name=Qwen2.5-VL-7B-Instruct_with_pose_encoder
output_dir=checkpoints/${run_name}

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$RDZV_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    internnav/trainer/s2_trainer_with_poses.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb