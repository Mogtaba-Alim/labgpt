#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Llama 3.1 8B model on code and research paper data.
This script handles diverse data formats and optimizes for high-quality fine-tuning
with an 8192 token context window.
"""

import os
import json
import logging
import random
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed_value: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    logger.info(f"Random seed set to {seed_value}")


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank for LoRA adaptation"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_nested_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use nested quantization for 4-bit (double quantization)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit quantization (fp4 or nf4)"}
    )
    use_flash_attn: bool = field(
        default=True, 
        metadata={"help": "Whether to use flash attention for faster training"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_file: str = field(
        default="combined_instruct_train.jsonl",
        metadata={"help": "Path to training data file (JSONL format with messages)"}
    )
    val_file: str = field(
        default="",
        metadata={"help": "Path to validation data file (JSONL format with messages, optional)"}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for the model"}
    )
    add_special_tokens: bool = field(
        default=True, 
        metadata={"help": "Whether to add special tokens to the data"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for training configuration."""
    output_dir: str = field(
        default="./llama-3.1-8b-finetuned",
        metadata={"help": "Output directory for model and checkpoints"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of updates steps to accumulate before backward pass"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay applied to parameters"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule type"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 mixed precision training"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 mixed precision training"}
    )
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "Optimizer to use for training"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints to save space"}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "The integration to report the results and logs to"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 on code and research paper data")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Detect GPU capabilities for bf16 and flash attention
    has_cuda = torch.cuda.is_available()
    cc_major = torch.cuda.get_device_capability(0)[0] if has_cuda else 0
    use_bf16 = (cc_major >= 8)  # Ampere+ (A100, A6000, RTX 30/40 series) supports bf16 well
    logger.info(f"CUDA available: {has_cuda}, Compute capability: {cc_major}, Using bf16: {use_bf16}")

    # Set up 4-bit quantization configuration if enabled
    compute_dtype = torch.bfloat16 if use_bf16 else getattr(torch, model_args.bnb_4bit_compute_dtype)
    
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )
    
    # Load model with appropriate configuration
    attn_impl = "flash_attention_2" if (model_args.use_flash_attn and has_cuda) else "eager"
    logger.info(f"Using attention implementation: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        use_cache=False,  # Gradient checkpointing requires this
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=True,
    )
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training if using LoRA
    if model_args.use_lora:
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    logger.info("Model and tokenizer loaded successfully")

    # Prepare datasets (JSONL with {"messages":[...], "metadata":{...}})
    logger.info(f"Loading JSONL train from: {data_args.train_file}")
    train_ds = load_dataset("json", data_files=data_args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=data_args.val_file, split="train") if data_args.val_file else None

    logger.info(f"Loaded {len(train_ds)} training examples")
    if eval_ds:
        logger.info(f"Loaded {len(eval_ds)} validation examples")

    # Map messages -> single "text" using official chat template
    def to_text(batch):
        texts = []
        for msgs in batch["messages"]:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False  # we include assistant outputs in SFT
            )
            texts.append(text)
        return {"text": texts}

    logger.info("Applying chat template to training data...")
    train_ds = train_ds.map(to_text, batched=True, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        logger.info("Applying chat template to validation data...")
        eval_ds = eval_ds.map(to_text, batched=True, remove_columns=eval_ds.column_names)

    logger.info(f"Processed {len(train_ds)} training examples with chat template")
    
    # Create the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=training_args.output_dir,
            max_seq_length=data_args.max_seq_length,
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            learning_rate=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            num_train_epochs=training_args.num_train_epochs,
            warmup_ratio=training_args.warmup_ratio,
            lr_scheduler_type=training_args.lr_scheduler_type,
            gradient_checkpointing=training_args.gradient_checkpointing,
            bf16=use_bf16,
            fp16=not use_bf16,
            optim=training_args.optim,
            logging_steps=training_args.logging_steps,
            eval_steps=max(200, training_args.eval_steps),
            save_steps=max(200, training_args.save_steps),
            save_total_limit=training_args.save_total_limit,
            report_to=training_args.report_to,
            seed=training_args.seed,
            packing=True,                 # Enable packing for better throughput (set False if most samples are long ~>4k)
            dataset_text_field="text",
            dataset_num_proc=1,
        ),
    )

    
    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    # Upload the model and tokenizer to the Hugging Face Hub
    logger.info("Uploading model to Hugging Face Hub")
    model.push_to_hub("llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba")
    tokenizer.push_to_hub("llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()