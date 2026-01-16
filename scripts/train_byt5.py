import argparse
import logging
import os
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Robust import for DataCollatorForSpanCorruption
try:
    from transformers import DataCollatorForSpanCorruption
except ImportError:
    try:
        from transformers.data.data_collator import DataCollatorForSpanCorruption
    except ImportError:
        print("WARNING: DataCollatorForSpanCorruption not found in this environment. Using DataCollatorForLanguageModeling as fallback.")
        from transformers import DataCollatorForLanguageModeling
        DataCollatorForSpanCorruption = None

def train(args):
    # 1. Load Tokenizer (Byt5)
    model_name = args.model_name
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load Data
    print(f"Loading data from {args.data_file}...")
    # Loading as text dataset
    dataset = load_dataset("text", data_files={"train": args.data_file})
    
    # 3. Preprocessing
    max_seq_length = args.max_seq_length
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    print("Tokenizing data...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Running tokenizer on dataset",
    )

    # 4. Data Collator (Span Corruption for T5/Byt5)
    if DataCollatorForSpanCorruption is not None:
        data_collator = DataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=0.15,
            mean_noise_span_length=3.0,
        )
    else:
        # Fallback for smoke test / environments without span corruption support
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False, 
        )

    # 5. Model
    print(f"Loading model {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=args.learning_rate,
        remove_unused_columns=False, # Important for DataCollatorForSpanCorruption
        # MPS support (Mac)
        use_mps_device=args.use_mps, 
        fp16=False, # MPS doesn't support fp16 mixed precision well yet usually
        push_to_hub=False,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/byt5-small", help="Model checkpoint")
    parser.add_argument("--data_file", type=str, required=True, help="Path to text data file")
    parser.add_argument("--output_dir", type=str, default="models/byt5-latin", help="Output directory")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--use_mps", action="store_true", help="Use MPS (Metal Performance Shaders) for Mac acceleration")
    
    args = parser.parse_args()
    
    # Check for MPS availability logic could be added here
    train(args)
