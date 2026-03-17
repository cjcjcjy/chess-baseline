import os
os.environ["WANDB_MODE"] = "disabled"

# Neuron/XLA optimizations - set before importing torch
os.makedirs("./neuron_cache", exist_ok=True)
os.environ["NEURON_COMPILE_CACHE_URL"] = "./neuron_cache"
os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --auto-cast=none"
os.environ["XLA_USE_BF16"] = "1"
# os.environ["NEURON_FUSE_SOFTMAX"] = "1"
# os.environ["XLA_DOWNCAST_BF16"] = "1"
# os.environ["NEURON_CC_PIPELINE_SIZE"] = "4"

import torch
import argparse
from datasets import Dataset
import wandb
from transformers import AutoTokenizer
from optimum.neuron.trainers import NeuronSFTTrainer, NeuronSFTConfig, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune LLM on chess dataset")
parser.add_argument(
    "--output-dir",
    type=str,
    default="./trained_models/chess_qwen_finetuned",
    help="Directory to save the fine-tuned model and checkpoints"
)
args = parser.parse_args()

# Validate output directory argument
assert args.output_dir and args.output_dir.strip(), "Output directory must be provided and non-empty"
if os.path.exists(args.output_dir):
    assert len(os.listdir(args.output_dir)) == 0, f"Output directory '{args.output_dir}' already exists and is not empty"

# Configuration
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_PATH = "./data/ChessExplained_2500k_qwen3.parquet"
assert os.path.exists(DATASET_PATH), f"Dataset file '{DATASET_PATH}' does not exist, please run the download script in the data directory"
TOKENIZER_PATH = "./data_preparation/chess_tokenizer_qwen3/"
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)
OUTPUT_DIR = args.output_dir
NUM_LINES_TO_LOAD = 1_000_0

MAX_LENGTH = 512  

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
WARMUP_STEPS = 500

# Logging and checkpointing
EVAL_STEPS = 3000
LOGGING_STEPS = EVAL_STEPS 
SAVE_STEPS = EVAL_STEPS
SAVE_TOTAL_LIMIT = 3

# Trainium-specific settings for trn1.2xlarge (2 NeuronCores)
TENSOR_PARALLEL_SIZE = 2

# Extract directory name for wandb run name
run_name = os.path.basename(os.path.normpath(OUTPUT_DIR))
wandb.init(project="ChessLLM", name=run_name)

print("="*80)
print("Training on AWS Trainium trn1.2xlarge")
print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Neuron Cache: {os.environ['NEURON_COMPILE_CACHE_URL']}")
print("="*80)

# %%
dataset = Dataset.from_parquet(DATASET_PATH).select(range(NUM_LINES_TO_LOAD))
print(f"Loaded {len(dataset)} examples")

# %%
# Load tokenizer
print(f"Loading tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Set pad token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

training_args = NeuronTrainingArguments(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    per_device_train_batch_size=BATCH_SIZE,  # Batch size per device
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    output_dir=OUTPUT_DIR,
    warmup_steps=WARMUP_STEPS,
    logging_steps=100,
    report_to="wandb",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    save_strategy="steps",
    dataloader_num_workers=4,
    dataloader_drop_last=True,
)

# Prepare model with resized embeddings
print(f"Loading model: {MODEL_NAME}")
print("Preparing model with resized embeddings...")

# Create a local directory for the prepared model
model_name_safe = MODEL_NAME.replace("/", "_")
PREPARED_MODEL_DIR = f"./{model_name_safe}_with_special_tokens"
if not os.path.exists(PREPARED_MODEL_DIR):
    print("Loading base model and resizing embeddings...")
    from transformers import AutoModelForCausalLM
    
    # Load model normally on CPU (will be moved to Neuron cores later)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    
    # Resize embeddings to match tokenizer
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Save both model and tokenizer to local directory
    print(f"Saving prepared model to {PREPARED_MODEL_DIR}")
    base_model.save_pretrained(PREPARED_MODEL_DIR)
    tokenizer.save_pretrained(PREPARED_MODEL_DIR)
    
    # Clean up base model to free memory
    del base_model
    print("Prepared model saved successfully!")
else:
    print(f"Using existing prepared model from {PREPARED_MODEL_DIR}")

# Load model optimized for Trainium from prepared directory
print("Loading model for Trainium training...")
print("Note: First compilation will take several minutes, subsequent runs will be fast!")

tokenizer = AutoTokenizer.from_pretrained(PREPARED_MODEL_DIR)
model = NeuronModelForCausalLM.from_pretrained(
    PREPARED_MODEL_DIR,
    training_args.trn_config,
    dtype=torch.bfloat16,
)

# Configure training for Trainium (needed before loading model)
training_config = NeuronSFTConfig(
    packing=True,
    **training_args.to_dict(),
)


print(f"Model loaded with {model.num_parameters():,} parameters")

# %%
# Formatting function for chess dataset
def format_chess_dataset(example):
    """Format chess dataset - simply return the text field."""
    return example["text"]

print("\n" + "="*80)
print("Training Configuration:")
print("="*80)
print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Weight decay: {training_args.weight_decay}")
print(f"  Warmup steps: {training_args.warmup_steps}")
print(f"  Total epochs: {training_args.num_train_epochs}")
print(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
print(f"  ZeRO-1 enabled: {training_args.zero_1}")
print(f"  Max gradient norm: {training_args.max_grad_norm}")
print("="*80 + "\n")

# %%
# Initialize NeuronSFTTrainer (following example.py pattern)
trainer = NeuronSFTTrainer(
    model=model,
    args=training_config,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=format_chess_dataset,
)

print("NeuronTrainer initialized. Ready to start training.")
print("\n⚠️  IMPORTANT: First run will compile graphs for a long time. Be patient!")
print("Subsequent runs will use cached compilation and be much faster.\n")

# %%
try:
    # Start training
    print("Starting training...")
    print("Monitor Neuron utilization with: neuron-top\n")

    trainer.train()

    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80 + "\n")

    # %%
    # Save the final model and tokenizer
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"Saving final model to {final_model_path}")

    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"\n✅ Model and tokenizer saved to {final_model_path}")
    print(f"\nTo load the model:")
    print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{final_model_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{final_model_path}")')
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Saving current checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted_checkpoint")
    print("Checkpoint saved.")
    
finally:
    wandb.finish()
    print("\nTraining session ended.")