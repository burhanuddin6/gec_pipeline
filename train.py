# CELL 2
import pandas as pd
from pathlib import Path
import torch

# CELL 3
def read_paired_files(incorrect_file, correct_file):
    """
    Read pairs from separate incorrect and correct files.
    Each line in incorrect_file corresponds to the same line in correct_file.
    """
    pairs = []
    
    # Define the prefix
    prefix = "Correct the grammar of the following Urdu sentence: "
    
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
    
    return pairs

# Update file paths to your actual files
incorrect_file_path = "data/out/shuffled_sampled_incorrect.txt"
correct_file_path = "data/out/shuffled_sampled_correct.txt"

print(f"Reading from:")
print(f"  Incorrect: {incorrect_file_path}")
print(f"  Correct: {correct_file_path}")

pairs = read_paired_files(incorrect_file_path, correct_file_path)
print(f"Loaded {len(pairs)} sentence pairs")

# Create dataset directly from pairs
from datasets import Dataset

dataset = Dataset.from_list(pairs)

# Split manually: first 20% for validation, remaining 80% for training
total_size = len(dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

validation_dataset = dataset.select(range(val_size))
train_dataset = dataset.select(range(val_size, total_size))

print(f"Validation dataset: {len(validation_dataset)} samples (first 20%)")
print(f"Train dataset: {len(train_dataset)} samples (remaining 80%)")

# --- Print a few examples from the VALIDATION set ---
print("--- First 3 Examples from the VALIDATION Set ---")
val_sample = validation_dataset.select(range(3))
for i, example in enumerate(val_sample):
    print(f"Example {i+1}:")
    print(f"Source (Incorrect): {example.get('source', 'N/A')}")
    print(f"Target (Correct):   {example.get('target', 'N/A')}")
    print("-" * 50)

print("\n" + "="*60 + "\n")

# --- Print a few examples from the TRAINING set ---
print("--- First 3 Examples from the TRAINING Set ---")
train_sample = train_dataset.select(range(3))
for i, example in enumerate(train_sample):
    print(f"Example {i+1}:")
    print(f"Source (Incorrect): {example.get('source', 'N/A')}")
    print(f"Target (Correct):   {example.get('target', 'N/A')}")
    print("-" * 50)

print("\n" + "="*60 + "\n")

# CELL 4
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
"""All your training code goes inside this function."""

model_name = "/home/sa07753/GEC/Model Procuring/mt0-large"
print("Loading tokenizer and model:", model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

 # --- SOLUTION 1: ENABLE GRADIENT CHECKPOINTING ---
# This is the most effective change you can make for memory. but it slows training down
# model.gradient_checkpointing_enable()

#CELL 5
import os

# This helps accelerate find the correct port for communication between processes
# os.environ["NCCL_P2P_DISABLE"] = "1"

# Assume 'dataset' is already loaded and available globally in your notebook
max_length = 128

def preprocess(batch):
    inputs = tokenizer(batch["source"], truncation=True, padding="max_length", max_length=max_length)
    targets = tokenizer(batch["target"], truncation=True, padding="max_length", max_length=max_length)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=["source", "target"])
tokenized_validation = validation_dataset.map(preprocess, batched=True, remove_columns=["source", "target"])
print("Tokenization complete.")

#CELL 6
per_device_batch = 12
class LoggingCallback(TrainerCallback):
    """A custom callback that prints logs to the console."""
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            print(logs)
my_logging_callback = LoggingCallback()

training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints/mt0/",
    save_strategy="steps",
    save_steps=63525, # 251400 (num_steps) / 4 to save 4 times per training
    eval_strategy="epoch",  # Add this
    # eval_steps=31761, # save_steps / 2
    per_device_train_batch_size=per_device_batch,
    per_device_eval_batch_size=per_device_batch,
#    gradient_accumulation_steps=8, 
    learning_rate=5e-6,
    num_train_epochs=3,
    weight_decay=0.01,  
    logging_strategy="steps",  
    logging_steps=10,
    save_total_limit=4,
    predict_with_generate=True,
    report_to="all",    # Add this to see logs in your console
    optim="adafactor"  # Use PyTorch's built-in Adafactor optimizer    
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,  # Now uses validation set    
    # The tokenizer argument is deprecated, but we can leave it for now
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[my_logging_callback]  # Add your custom logging callback
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("checkpoints/mt0/final/")