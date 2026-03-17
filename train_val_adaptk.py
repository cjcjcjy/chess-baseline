import os
# os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# nohup python train_val_adaptk.py --output-dir ./qwen3_4b_new_format_val_adaptk --resume-from /home/jcyang/Global-Chess-Challenge-2025-Baselines/qwen3_4b_new_format_val_adaptk/checkpoint-70000 > train_fmt_val_adaptk_from70000.log 2>&1 &

# huggingface-cli upload w007425y/xiangqi_sft /home/jcyang/Global-Chess-Challenge-2025-Baselines/qwen3_4b/checkpoint-4000 --commit-message "official ckpt4000"

import torch
import pandas as pd
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune LLM on chess dataset")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./trained_models/chess_qwen_finetuned",
    help="Directory to save the fine-tuned model and checkpoints"
)
parser.add_argument(
    "--resume-from",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from (e.g., ./qwen3_4b_new_format/checkpoint-34000)"
)
args = parser.parse_args()

# Validate output directory argument
assert args.output_dir and args.output_dir.strip(), "Output directory must be provided and non-empty"
if args.resume_from is None:
    # Only check for empty directory when not resuming
    if os.path.exists(args.output_dir):
        assert len(os.listdir(args.output_dir)) == 0, f"Output directory '{args.output_dir}' already exists and is not empty"
else:
    # Validate resume checkpoint path
    assert os.path.exists(args.resume_from), f"Resume checkpoint path '{args.resume_from}' does not exist"
    print(f"Will resume training from checkpoint: {args.resume_from}")

# Configuration
MODEL_NAME = "/home/jcyang/models/Qwen/Qwen3-4B"
DATASET_PATH = "/home/jcyang/Global-Chess-Challenge-2025-Baselines/data/ChessExplained_scored_obs_adaptk.parquet"
assert os.path.exists(DATASET_PATH), f"Dataset file '{DATASET_PATH}' does not exist, please run the download script in the data directory"
TOKENIZER_PATH = "./data_preparation/chess_tokenizer_qwen3/"
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)
OUTPUT_DIR = args.output_dir
NUM_LINES_TO_LOAD = 1543676
# Maximum sequence length, for the dataset with special tokens, the sequences are short so 512 is enough
MAX_LENGTH = 700
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
WARMUP_STEPS = 500
LOGGING_STEPS = 500
EVAL_STEPS = 1000  # 每 1000 步评估一次
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 100
NUM_TRAIN_EPOCHS = 1

# Extract directory name for run name
os.environ["SWANLAB_MODE"] = "local"
os.environ["SWANLAB_PROJECT"] = "ChessLLM"

print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# %%
dataset = Dataset.from_parquet(DATASET_PATH).select(range(NUM_LINES_TO_LOAD))
print(f"Loaded {len(dataset)} examples")

# %%
# Load tokenizer and model
print(f"Loading model: {MODEL_NAME}")
print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    # MODEL_NAME,
    TOKENIZER_PATH,
    trust_remote_code=True,
    padding_side="right"
)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    # attn_implementation="flash_attention_2",
)
model.resize_token_embeddings(len(tokenizer))

print(f"Model loaded with {model.num_parameters():,} parameters")

# %%
# Tokenize the dataset

import numpy as np

# -----------------------------------------------------------------------------
# 📊 Token 长度分析 & 过滤超长数据
# -----------------------------------------------------------------------------
print(f"\n[Analysis] Calculating token lengths statistics (MAX_LENGTH={MAX_LENGTH})...")

def get_raw_lengths(examples):
    # 使用 truncation=False 获取原始完整长度
    # 注意：这里只返回 length，不返回庞大的 input_ids 以节省内存
    out = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
        return_attention_mask=False
    )
    return {"raw_len": [len(ids) for ids in out["input_ids"]]}

# 1. 计算所有样本的原始长度
# 使用 map 处理，batched=True 速度更快
length_ds = dataset.map(
    get_raw_lengths,
    batched=True,
    batch_size=5000,
    desc="Analyzing Lengths"
)

raw_lengths = np.array(length_ds["raw_len"])
total_samples = len(raw_lengths)

# 2. 统计超长数据量
num_exceed = np.sum(raw_lengths > MAX_LENGTH)
percent_exceed = (num_exceed / total_samples) * 100

# 3. 打印文本统计报告
print(f"\n{'='*60}")
print(f"🔢 Tokenizer Length Statistics Report")
print(f"{'='*60}")
print(f"Total Samples       : {total_samples}")
print(f"Max Sequence Length : {MAX_LENGTH}")
print(f"Samples > MAX_LENGTH: {num_exceed} (⚠️ {percent_exceed:.2f}% will be filtered out)")
print(f"{'-'*60}")
print(f"{'Metric':<12} | {'Original (Raw)':<18}")
print(f"{'-'*60}")
print(f"{'Min':<12} | {np.min(raw_lengths):<18}")
print(f"{'Max':<12} | {np.max(raw_lengths):<18}")
print(f"{'Mean':<12} | {np.mean(raw_lengths):<18.2f}")
print(f"{'Median':<12} | {np.median(raw_lengths):<18.0f}")
print(f"{'95% Pctl':<12} | {np.percentile(raw_lengths, 95):<18.0f}")
print(f"{'99% Pctl':<12} | {np.percentile(raw_lengths, 99):<18.0f}")
print(f"{'='*60}\n")

# 4. 过滤掉超过 MAX_LENGTH 的数据（不截断）
print(f"Filtering out samples exceeding MAX_LENGTH={MAX_LENGTH}...")
dataset = length_ds.filter(
    lambda x: x["raw_len"] <= MAX_LENGTH,
    desc="Filtering"
)
# 移除 raw_len 列
dataset = dataset.remove_columns(["raw_len"])
print(f"After filtering: {len(dataset)} samples remaining ({total_samples - len(dataset)} removed)")

# 划分训练集和验证集
VALIDATION_SPLIT = 0.001  # 1% 作为验证集
print(f"\nSplitting dataset into train/validation sets (validation ratio: {VALIDATION_SPLIT*100:.1f}%)...")
split_dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"Train set: {len(train_dataset)} samples")
print(f"Validation set: {len(eval_dataset)} samples")

# -----------------------------------------------------------------------------

def tokenize_function(examples):
    """Tokenize text examples (no truncation since we already filtered)"""
    tokenized = tokenizer(
        examples["text"],
        truncation=False,  # 不截断，因为已经过滤掉超长数据
        padding=False,  # We'll pad dynamically during training
        return_tensors=None
    )
    return tokenized

print("Tokenizing train dataset...")
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing Train"
)

print("Tokenizing validation dataset...")
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing Validation"
)

print(f"Tokenization complete. Train sample token count: {len(tokenized_train_dataset[0]['input_ids'])}")
print(f"Validation sample token count: {len(tokenized_eval_dataset[0]['input_ids'])}")


# %%
# Set up training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name="new_format_val_obvious_adaptk",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # 评估时可以用更大的 batch
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=False,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,  # Add this line
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    eval_strategy="steps",  # 按步数评估
    eval_steps=EVAL_STEPS,  # 每 EVAL_STEPS 步评估一次
    bf16=True,
    fp16=False,
    optim="adamw_torch_fused",
    remove_unused_columns=False,
    report_to="swanlab",
    dataloader_drop_last=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
)

print("Training configuration:")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Eval steps: {training_args.eval_steps}")

# %%
# Create data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
)

print(f"Trainer initialized with validation set ({len(tokenized_eval_dataset)} samples). Ready to start training.")

# %%
try:
# Start training
    print("Starting training...")

    trainer.train(resume_from_checkpoint=args.resume_from)

    print("\n✅ Training complete!")

    # %%
    # Save the final model and tokenizer
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"Saving final model to {final_model_path}")

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"✅ Model and tokenizer saved to {final_model_path}")
    print(f"\nTo load the model:")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{final_model_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{final_model_path}")')
except KeyboardInterrupt:
    print("Training interrupted by user")
