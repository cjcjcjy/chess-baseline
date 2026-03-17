# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
set -x
# nohup bash run_qwen3-4b.sh > qwen3-4b-grpo-new_format.log 2>&1 &

export SWANLAB_MODE="local"
export NCCL_P2P_DISABLE=1  # 禁用OpenMP多线程/P2P
CUDA_VISIBLE_DEVICES=1,2 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/verl/examples/data_preprocess/grpo_filter200000/grpo_data_200000.parquet \
    data.val_files=$HOME/verl/examples/data_preprocess/chess_grpo_data/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HOME/Global-Chess-Challenge-2025-Baselines/qwen3_4b_new_format_val_adaptk_dp12/checkpoint-90000 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name='verl_grpo_chess' \
    trainer.experiment_name='qwen3_4b_grpo_new_format' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 $@ \
    trainer.max_actor_ckpt_to_keep=100 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.rollout.load_format='safetensors' 
    # actor_rollout_ref.rollout.temperature=1.0 \
    # actor_rollout_ref.rollout.top_p=0.95 \
    # actor_rollout_ref.rollout.top_k=50 \
    # actor_rollout_ref.rollout.stop_sequences='["<|im_end|>"]' \
    # actor_rollout_ref.rollout.val_kwargs.do_sample=True