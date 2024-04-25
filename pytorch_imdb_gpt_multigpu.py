#!/usr/bin/env python
# coding: utf-8

# # IMDB movie review text generation
#
# In this script, we'll fine-tune a GPT2-like model to generate more
# movie reviews based on a prompt.
#
# Partly based on this tutorial:
# https://github.com/omidiu/GPT-2-Fine-Tuning/

import torch
import os
import math
import argparse

from pprint import pprint
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, help="The root directory under which model checkpoints are stored.")
parser.add_argument("--model-name", type=str, default="gpt-imdb-model", help="A name to store the trained model under.")
parser.add_argument("--num_workers", type=int, default=1, help="The number of CPU worker processes to use.")
args, _ = parser.parse_known_args()

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

print('Using PyTorch version:', torch.__version__)
if torch.cuda.is_available():
    print(f"Rank {rank} of {world_size} (local: {local_rank}) sees {torch.cuda.device_count()} devices")
    print('Using GPU, device name:', torch.cuda.get_device_name(local_rank))
    device = torch.device(f"cuda:{local_rank}")
else:
    print('No GPU found, using CPU instead.')
    device = torch.device('cpu')

user_datapath = args.datadir
os.makedirs(user_datapath, exist_ok=True)


# ## IMDB data set
#
# Next we'll load the IMDB data set, this time using the Hugging Face
# datasets library: https://huggingface.co/docs/datasets/index.
#
# The dataset contains 100,000 movies reviews from the Internet Movie
# Database, split into 25,000 reviews for training and 25,000 reviews
# for testing and 50,000 without labels (unsupervised).

train_dataset = load_dataset("imdb", split="train+unsupervised", trust_remote_code=False, keep_in_memory=True)
eval_dataset = load_dataset("imdb", split="test", trust_remote_code=False, keep_in_memory=True)

# Let's print one sample from the dataset.
if rank == 0:
    print('Sample from dataset')
    for b in train_dataset:
        pprint(b)
        break

# We set up the training configuration here
output_dir = os.path.join(user_datapath, args.model_name)
logging_dir = os.path.join(user_datapath, "runs", args.model_name)
train_batch_size = 32
eval_batch_size = 128

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy="no",
    logging_dir=logging_dir,
    evaluation_strategy="steps",
    eval_steps=200,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    ddp_find_unused_parameters=False,
    per_device_train_batch_size=train_batch_size // world_size,
    per_device_eval_batch_size=eval_batch_size,
    max_steps=1000,
    dataloader_num_workers=args.num_workers,
    dataloader_pin_memory=True,
    report_to=["tensorboard"],
)


# We'll use the gpt-neo-1.3B model from the Hugging Face library:
# https://huggingface.co/EleutherAI/gpt-neo-1.3B
# Let's start with getting the appropriate tokenizer.
pretrained_model = "EleutherAI/gpt-neo-1.3B"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
special_tokens = tokenizer.special_tokens_map

# Load the actual base model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(pretrained_model)
model.to(device)


# We tokenize the text and add the special token for indicating the end of the
# text at the end of each review. We also truncate reviews to a maximum
# length to avoid excessively long sequences during training.
max_length=256
def tokenize(x):
    texts = [example + tokenizer.eos_token for example in x['text']]
    return tokenizer(texts, max_length=max_length, truncation=True, add_special_tokens=True, return_overflowing_tokens=True, return_length=False)

train_dataset_tok = train_dataset.map(tokenize,
                                      remove_columns=['text', 'label'],
                                      batched=True,
                                      batch_size=training_args.train_batch_size,
                                      num_proc=training_args.dataloader_num_workers)

eval_dataset_tok = eval_dataset.map(tokenize,
                                    remove_columns=['text', 'label'],
                                    batched=True,
                                    num_proc=training_args.dataloader_num_workers)


# we split a small amount of data as "validation" test set to keep track of evoluation of the loss on non-training data during training
# this is purely because computing the loss on the full evaluation dataset takes much longer
train_validate_splits = train_dataset_tok.train_test_split(test_size=1000, seed=42, keep_in_memory=True)
train_dataset_tok = train_validate_splits['train']
validate_dataset_tok = train_validate_splits['test']

if rank == 0:
    print('Sample of tokenized data')
    for b in train_dataset_tok:
        pprint(b, compact=True)
        print('Length of input_ids:', len(b['input_ids']))
        break
    print('Length of dataset (tokenized)', len(train_dataset_tok))

    num_batches = len(train_dataset) // train_batch_size
    print(f"Training set is comprised of {num_batches} batches")


# Here we use the Hugging Face trainer instead of our own training
# function

# You can read about the many, many different parameters to the
# Hugging Face trainer here:
# https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors='pt')

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=collator,
    train_dataset=train_dataset_tok,
    eval_dataset=validate_dataset_tok,
)

trainer.train()

if rank == 0:
    print()
    print("Training done, you can find all the model checkpoints in", output_dir)

with torch.no_grad():
    # Calculate perplexity
    eval_results = trainer.evaluate()
    test_results = trainer.evaluate(eval_dataset_tok)

    if rank == 0:
        print(f'Perplexity on validation: {math.exp(eval_results["eval_loss"]):.2f}')
        print(f'Perplexity on test: {math.exp(test_results["eval_loss"]):.2f}')

        # Let's print a few sample generated reviews
        prompt = "The movie 'How to run ML on LUMI - A documentation' was great because"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        outputs = model.generate(**inputs, do_sample=True, max_length=80, num_return_sequences=4)
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print('Sample generated review:')
        for txt in decoded_outputs:
            print('-', txt)
