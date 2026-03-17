set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen3_8b_sft_peft_sp2_npu.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

# nproc_per_node=$1
export OMP_NUM_THREADS=1
# export NCCL_P2P_DISABLE=1

# CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash run_qwen3_4b_sft.sh > chess_sft_qwen3_4b_forchess.log 2>&1 &
# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=640 \
    data.train_files=$HOME/global-chess-challenge-2025-starter-kit/data/chess_sft_2500k_512/train.parquet \
    data.val_files=$HOME/global-chess-challenge-2025-starter-kit/data/chess_sft_2500k_512/test.parquet \
    data.prompt_key=prompt \
    data.response_key=solution \
    data.max_length=512 \
    optim.lr=5e-5 \
    data.micro_batch_size_per_gpu=32 \
    model.partial_pretrain=$HOME/models/Qwen/Qwen3-4B-forchess \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    trainer.project_name=chess-sft \
    trainer.experiment_name=chess-sft-qwen3-4b-forchess \
    trainer.logger='["console","swanlab"]' \
    trainer.save_freq=200 \
    trainer.test_freq=200 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.debug_print_freq=50 \
    model.strategy=fsdp \
    model.lora_rank=32 \
    model.lora_alpha=64 \
    model.target_modules=all-linear \
    trainer.checkpoint.save_contents=[model,optimizer,extra] \
    +model.fsdp_config.model_type=bfloat16 \
     # trainer.resume_mode=resume_path \
     # trainer.resume_from_path=/home/jcyang/verl/examples/sft/chess/checkpoints/chess-sft/chess-sft-qwen3-4b/global_step_6800

    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=true \

    # trainer.resume_mode=disable \
    # trainer.resume_mode=resume_path \
    # trainer.resume_from_path=~/checkpoints/gsm8k-sft/qwen/global_step_100

