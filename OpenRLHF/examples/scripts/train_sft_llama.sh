set -x
export CUDA_VISIBLE_DEVICES=0 \
export WANDB_API_KEY='0b072f51f17c57752abce89f82d0274da6ced867'
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset path2data \
   --input_key prompt \
   --output_key chosen_response_only \
   --train_batch_size 64 \
   --micro_train_batch_size 16 \
   --max_samples 500000 \
   --pretrain path2llama \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --adam_offload \
   --overlap_comm \
   --use_wandb 0b072f51f17c57752abce89f82d0274da6ced867 
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi