set -x
export CUDA_VISIBLE_DEVICES=0 
export WANDB_API_KEY='0b072f51f17c57752abce89f82d0274da6ced867'
read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3-8b-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --dataset path2data \
   --micro_train_batch_size 16 \
   --pretrain path2llama \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing \
   --adam_offload \
   --overlap_comm \
   --use_wandb 0b072f51f17c57752abce89f82d0274da6ced867 \

EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
