"""
Modal GPU Training Script for Chess LLM

Usage:
1. First, upload the dataset and tokenizer to Modal volumes:
   modal run train_modal.py::upload_data

2. Then, run the training:
   modal run train_modal.py::train

3. Or run with a specific GPU:
   modal run train_modal.py::train --gpu a100-80gb

4. Download the trained model:
   modal run train_modal.py::download_model
"""

import os
import modal

# Create Modal app
app = modal.App("chess-llm-training")

# Create persistent volumes for data and model storage
data_volume = modal.Volume.from_name("chess-data", create_if_missing=True)
model_volume = modal.Volume.from_name("chess-models", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.0",
        "datasets==2.20.0",
        "accelerate==0.33.0",
        "pandas==2.2.2",
        "pyarrow==17.0.0",
        "wandb==0.17.0",
        "swanlab==0.3.0",
        "huggingface_hub>=0.24.0",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
)


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    timeout=3600,
)
def upload_data():
    """Upload dataset and tokenizer to Modal volume"""
    import shutil
    from pathlib import Path
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Downloading dataset from HuggingFace...")
    # You may need to upload your dataset to HuggingFace or use a different method
    # For now, we assume the data will be uploaded manually or exists

    # Create directories
    Path("/data/tokenizer").mkdir(parents=True, exist_ok=True)
    Path("/data/dataset").mkdir(parents=True, exist_ok=True)

    data_volume.commit()
    print("Data directories created. Please upload your data using:")
    print("  - Dataset: /data/dataset/ChessExplained_scored.parquet")
    print("  - Tokenizer: /data/tokenizer/")


@app.function(
    image=image,
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    timeout=300,
)
def upload_local_files():
    """Check what files exist in volumes"""
    import os

    print("=== Data Volume Contents ===")
    for root, dirs, files in os.walk("/data"):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"  {filepath} ({size:.2f} MB)")

    print("\n=== Models Volume Contents ===")
    for root, dirs, files in os.walk("/models"):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"  {filepath} ({size:.2f} MB)")


@app.function(
    image=image,
    gpu="A100",  # Options: "T4", "A10G", "A100", "A100-80GB", "H100"
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    timeout=86400,  # 24 hours
    memory=32768,   # 32GB RAM
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def train(
    model_name: str = "Qwen/Qwen3-4B",  # Use HF model name, matches original script
    num_lines: int = 700000,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    max_length: int = 576,
    output_name: str = "chess_qwen_finetuned",
    use_wandb: bool = False,
    wandb_project: str = "chess-llm",
    warmup_steps: int = 500,
    save_steps: int = 500,
):
    """Main training function that runs on Modal GPU"""
    import torch
    import pandas as pd
    from pathlib import Path
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )

    # Configuration - paths match the upload script
    DATASET_PATH = "/data/dataset/ChessExplained_scored.parquet"
    TOKENIZER_PATH = "/data/tokenizer"
    OUTPUT_DIR = f"/models/{output_name}"

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please upload the dataset first using modal volume put."
        )

    # Set up wandb if enabled
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, name=output_name)
        report_to = "wandb"
    else:
        os.environ["WANDB_MODE"] = "disabled"
        report_to = "none"

    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}")
    dataset = Dataset.from_parquet(DATASET_PATH)
    if num_lines and num_lines < len(dataset):
        dataset = dataset.select(range(num_lines))
    print(f"Loaded {len(dataset)} examples")

    # Load tokenizer
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    if os.path.exists(TOKENIZER_PATH) and os.listdir(TOKENIZER_PATH):
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH,
            trust_remote_code=True,
            padding_side="right"
        )
    else:
        print(f"Custom tokenizer not found, using model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model loaded with {model.num_parameters():,} parameters")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        num_proc=4,
    )
    print(f"Tokenization complete. Sample token count: {len(tokenized_dataset[0]['input_ids'])}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=output_name,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,  # Save memory on single GPU
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_steps=warmup_steps,
        logging_steps=100,
        save_steps=save_steps,
        save_total_limit=5,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        report_to=report_to,
        dataloader_drop_last=True,
        num_train_epochs=num_epochs,
        dataloader_num_workers=4,
        logging_first_step=True,
    )

    print("Training configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {grad_accum_steps}")
    print(f"  Effective batch size: {batch_size * grad_accum_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Num epochs: {num_epochs}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save final model
    final_model_path = f"{OUTPUT_DIR}/final_model"
    print(f"Saving final model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # Commit volumes
    model_volume.commit()

    print(f"Model saved to {final_model_path}")
    return {"status": "success", "model_path": final_model_path}


@app.function(
    image=image,
    volumes={
        "/models": model_volume,
    },
    timeout=3600,
)
def download_model(model_name: str = "chess_qwen_finetuned"):
    """List and prepare model for download"""
    from pathlib import Path
    import shutil

    model_path = Path(f"/models/{model_name}/final_model")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Available models:")
        for d in Path("/models").iterdir():
            print(f"  - {d.name}")
        return

    print(f"Model found at {model_path}")
    print("Files:")
    for f in model_path.iterdir():
        size = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size:.2f} MB")

    print("\nTo download the model, use:")
    print(f"  modal volume get chess-models {model_name}/final_model ./local_model")


@app.local_entrypoint()
def main(
    action: str = "train",
    model_name: str = "Qwen/Qwen3-4B",
    num_lines: int = 700000,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 1e-4,
    num_epochs: int = 1,
    output_name: str = "chess_qwen_finetuned",
    use_wandb: bool = False,
    warmup_steps: int = 500,
    save_steps: int = 500,
):
    """
    Main entrypoint for the training script

    Actions:
        upload  - Create Modal volumes for data storage
        check   - Check contents of Modal volumes
        train   - Run training on Modal GPU
        download - Show how to download trained model

    Examples:
        modal run train_modal.py --action train
        modal run train_modal.py --action train --num-lines 100000 --batch-size 8
        modal run train_modal.py --action check
    """
    if action == "upload":
        upload_data.remote()
    elif action == "check":
        upload_local_files.remote()
    elif action == "train":
        result = train.remote(
            model_name=model_name,
            num_lines=num_lines,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            output_name=output_name,
            use_wandb=use_wandb,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
        )
        print(f"Training result: {result}")
    elif action == "download":
        download_model.remote(output_name)
    else:
        print(f"Unknown action: {action}")
        print("Available actions: upload, check, train, download")
