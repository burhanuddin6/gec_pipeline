"""
Clean Training Script for Urdu Grammar Error Correction
Features:
- Automatic caching of tokenized datasets
- Checkpoint resume functionality
- Prevents redundant recalculation
- Clear logging of what's being loaded vs computed
"""

import pandas as pd
from pathlib import Path
import torch
import os
import sys
import glob
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq, 
    TrainerCallback, 
    TrainingArguments, 
    TrainerState, 
    TrainerControl
)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_name": "bigscience/mt0-small",
    "incorrect_file": "data/out/shrink_incorrect.txt",
    "correct_file": "data/out/shrink_correct.txt",
    "checkpoint_dir": "training/checkpoints_mt0_small/",
    "cache_dir": "training/tokenized_cache_mt0_small",
    "max_length": 128,
    "prefix": "اس جملے کی گرامر درست کریں: ",
    "validation_split": 0.1,
    "per_device_batch": 8,
    "learning_rate": 5e-6,
    "num_train_epochs": 8,
    "save_steps": 1000,
    "logging_steps": 500,
    "restart_after_steps": 2000,
}

# ============================================================================
# Checkpoint Management
# ============================================================================

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the output directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # Sort by step number and get the latest
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print(f"✓ Found checkpoint: {latest}")
    return latest

# ============================================================================
# Dataset Loading and Caching
# ============================================================================

def check_tokenized_cache_exists(cache_dir):
    """Check if tokenized datasets exist in cache"""
    train_path = os.path.join(cache_dir, "tokenized_train")
    val_path = os.path.join(cache_dir, "tokenized_validation")
    
    exists = os.path.exists(train_path) and os.path.exists(val_path)
    if exists:
        print(f"✓ Found cached tokenized datasets in {cache_dir}")
    else:
        print(f"✗ No cached tokenized datasets found in {cache_dir}")
    return exists

def load_tokenized_datasets(cache_dir):
    """Load tokenized datasets from disk"""
    train_path = os.path.join(cache_dir, "tokenized_train")
    val_path = os.path.join(cache_dir, "tokenized_validation")
    
    print(f"Loading tokenized datasets from {cache_dir}...")
    tokenized_train = Dataset.load_from_disk(train_path)
    tokenized_validation = Dataset.load_from_disk(val_path)
    print(f"✓ Loaded {len(tokenized_train)} training samples")
    print(f"✓ Loaded {len(tokenized_validation)} validation samples")
    return tokenized_train, tokenized_validation

def save_tokenized_datasets(tokenized_train, tokenized_validation, cache_dir):
    """Save tokenized datasets to disk"""
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "tokenized_train")
    val_path = os.path.join(cache_dir, "tokenized_validation")
    
    print(f"Saving tokenized datasets to {cache_dir}...")
    tokenized_train.save_to_disk(train_path)
    tokenized_validation.save_to_disk(val_path)
    print(f"✓ Saved {len(tokenized_train)} training samples")
    print(f"✓ Saved {len(tokenized_validation)} validation samples")

def read_paired_files(incorrect_file, correct_file, prefix):
    """
    Read pairs from separate incorrect and correct files.
    Each line in incorrect_file corresponds to the same line in correct_file.
    """
    print(f"Reading data from:")
    print(f"  Incorrect: {incorrect_file}")
    print(f"  Correct: {correct_file}")
    
    if not os.path.exists(incorrect_file):
        raise FileNotFoundError(f"Incorrect file not found: {incorrect_file}")
    if not os.path.exists(correct_file):
        raise FileNotFoundError(f"Correct file not found: {correct_file}")
    
    pairs = []
    
    with open(incorrect_file, 'r', encoding='utf-8') as f_incorrect, \
         open(correct_file, 'r', encoding='utf-8') as f_correct:
        
        incorrect_lines = [line.strip() for line in f_incorrect if line.strip()]
        correct_lines = [line.strip() for line in f_correct if line.strip()]
    
    # Ensure both files have the same number of lines
    if len(incorrect_lines) != len(correct_lines):
        raise ValueError(f"Files have different lengths: {len(incorrect_lines)} vs {len(correct_lines)}")
    
    # Create pairs with prefix (source=prefix+incorrect, target=correct)
    for incorrect, correct in zip(incorrect_lines, correct_lines):
        pairs.append({"source": prefix + incorrect, "target": correct})
    
    print(f"✓ Loaded {len(pairs)} sentence pairs")
    return pairs

def create_datasets_from_files(incorrect_file, correct_file, prefix, validation_split):
    """Create train and validation datasets from raw files"""
    pairs = read_paired_files(incorrect_file, correct_file, prefix)
    
    # Create dataset directly from pairs
    dataset = Dataset.from_list(pairs)
    
    # Split: first validation_split% for validation, remaining for training
    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    
    validation_dataset = dataset.select(range(val_size))
    train_dataset = dataset.select(range(val_size, total_size))
    
    print(f"✓ Validation dataset: {len(validation_dataset)} samples ({validation_split*100}%)")
    print(f"✓ Train dataset: {len(train_dataset)} samples ({(1-validation_split)*100}%)")
    
    return train_dataset, validation_dataset

def tokenize_datasets(train_dataset, validation_dataset, tokenizer, max_length):
    """Tokenize train and validation datasets"""
    print("Tokenizing datasets...")
    
    def preprocess(batch):
        inputs = tokenizer(batch["source"], truncation=True, padding="max_length", max_length=max_length)
        targets = tokenizer(batch["target"], truncation=True, padding="max_length", max_length=max_length)
        inputs["labels"] = targets["input_ids"]
        return inputs
    
    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=["source", "target"])
    tokenized_validation = validation_dataset.map(preprocess, batched=True, remove_columns=["source", "target"])
    
    print(f"✓ Tokenization complete")
    return tokenized_train, tokenized_validation

# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(model_name):
    """Load tokenizer and model"""
    print(f"Loading model and tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    
    print("✓ Model and tokenizer loaded")
    return tokenizer, model

# ============================================================================
# Training Callbacks
# ============================================================================

class RestartAfterSaveCallback(TrainerCallback):
    """Callback that stops training after saving and triggers restart"""
    def __init__(self, restart_after_steps):
        self.restart_after_steps = restart_after_steps
        self.start_step = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_step = state.global_step
        print(f"Training session starting at step {self.start_step}")
    
    def on_save(self, args, state, control, **kwargs):
        if self.start_step is not None:
            steps_this_session = state.global_step - self.start_step
            print(f"Checkpoint saved at step {state.global_step} ({steps_this_session} steps this session)")
            
            if steps_this_session >= self.restart_after_steps:
                print(f"Completed {steps_this_session} steps. Stopping training to restart...")
                control.should_training_stop = True

class LoggingCallback(TrainerCallback):
    """A custom callback that prints logs to the console."""
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            print(logs)

def restart_script_in_new_terminal():
    """Restart the script in the same terminal"""
    print("=" * 60)
    print("RESTARTING TRAINING SESSION")
    print("=" * 60)
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    os.execl(python_path, python_path, *sys.argv)

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function"""
    print("=" * 60)
    print("URDU GRAMMAR ERROR CORRECTION - TRAINING")
    print("=" * 60)
    
    # Step 1: Check for cached tokenized datasets
    print("\n[1/5] Checking for cached tokenized datasets...")
    cache_exists = check_tokenized_cache_exists(CONFIG["cache_dir"])
    
    if cache_exists:
        # Load cached tokenized datasets
        tokenized_train, tokenized_validation = load_tokenized_datasets(CONFIG["cache_dir"])
        
        # Load model and tokenizer
        print("\n[2/5] Loading model and tokenizer...")
        tokenizer, model = load_model_and_tokenizer(CONFIG["model_name"])
        
    else:
        # No cache - need to create datasets from scratch
        print("\n[2/5] Creating datasets from raw files...")
        train_dataset, validation_dataset = create_datasets_from_files(
            CONFIG["incorrect_file"],
            CONFIG["correct_file"],
            CONFIG["prefix"],
            CONFIG["validation_split"]
        )
        
        # Load model and tokenizer
        print("\n[3/5] Loading model and tokenizer...")
        tokenizer, model = load_model_and_tokenizer(CONFIG["model_name"])
        
        # Tokenize datasets
        print("\n[4/5] Tokenizing datasets...")
        tokenized_train, tokenized_validation = tokenize_datasets(
            train_dataset,
            validation_dataset,
            tokenizer,
            CONFIG["max_length"]
        )
        
        # Save tokenized datasets for future use
        save_tokenized_datasets(tokenized_train, tokenized_validation, CONFIG["cache_dir"])
    
    # Step 2: Setup training
    print("\n[5/5] Setting up training...")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    
    # Setup callbacks
    restart_callback = RestartAfterSaveCallback(restart_after_steps=CONFIG["restart_after_steps"])
    logging_callback = LoggingCallback()
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=CONFIG["checkpoint_dir"],
        save_strategy="steps",
        save_steps=CONFIG["save_steps"],
        per_device_train_batch_size=CONFIG["per_device_batch"],
        per_device_eval_batch_size=CONFIG["per_device_batch"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=CONFIG["logging_steps"],
        save_total_limit=4,
        predict_with_generate=True,
        report_to="all",
        optim="adamw_torch",

        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        group_by_length=True,
        gradient_accumulation_steps=4
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[restart_callback, logging_callback]
    )
    
    # Check for existing checkpoint
    print("\nChecking for existing checkpoints...")
    latest_checkpoint = get_latest_checkpoint(CONFIG["checkpoint_dir"])
    
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        resume_from = latest_checkpoint
    else:
        print("No checkpoint found, starting from scratch")
        resume_from = None
    
    # Start training
    print(f"\nStarting training session (PID: {os.getpid()})...")
    print("=" * 60)
    
    # Run training
    trainer.train(resume_from_checkpoint=resume_from)
    
    # === FIX STARTS HERE ===
    
    # Get the current status
    current_step = trainer.state.global_step
    max_steps = trainer.state.max_steps

    # Logic: Only restart if we haven't reached the finish line yet
    if current_step >= max_steps:
        print("\n" + "="*60)
        print(f"✓ TRAINING COMPLETE ({current_step}/{max_steps} steps)")
        print("Exiting script safely.")
        print("="*60)
        sys.exit(0)  # Stop the script completely
    else:
        # If we are here, it means we stopped early (due to memory cleaning callback)
        print(f"\nPaused at step {current_step}/{max_steps}. Restarting for memory cleanup...")
        restart_script_in_new_terminal()

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()