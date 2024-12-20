# Multi-nodes are also supported

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp83s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export DS_SKIP_CUDA_CHECK=1

export MASTER_ADDR="${CHIEF_IP:=localhost}"
export MASTER_PORT="${MASTER_PORT:=31600}"
export HOST_NUM=1
export INDEX=0


wandb offline

train_path=transformers/examples/pytorch/language-modeling/chain_of_train.py
model_path=""
model_save=""
sim_model=""
train_file=""



# HOST_NUM will be 1
torchrun --nnodes $HOST_NUM --node_rank $INDEX --nproc_per_node 4 --master_addr $MASTER_ADDR --master_port $MASTER_PORT  \
    ${train_path} \
    --model_name_or_path ${model_path} \
    --deepspeed train/deepspeed_config_zero2.json \
    --train_file ${train_file} \
    --preprocessing_num_workers 1 \
    --dataloader_num_workers 1 \
    --dataloader_pin_memory True \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1.5 \
    --save_strategy "steps" \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --block_size 512 \
    --do_train \
    --evaluation_strategy "no" \
    --validation_split_percentage 1 \
    --fp16 True \
    --fp16_full_eval True \
    --ddp_timeout 3600 \
    --seed 1 \
    --gradient_checkpointing True \
    --output_dir ${model_save} \
    --lora_path "../graph/save_parrot/checkpoints/checkpoint_48/"

# Use streaming for large datasets and specify the max_steps
#    --streaming \
#    --max_steps 2500 \
