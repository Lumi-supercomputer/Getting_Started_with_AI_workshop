#!/usr/bin/env python
# coding: utf-8

# # IMDB movie review text generation
#
# In this script, we'll fine-tune a GPT2-like model to generate more
# movie reviews based on a prompt.
#
# Partly based on this tutorial:
# https://github.com/omidiu/GPT-2-Fine-Tuning/


import argparse
import math
import os
import time
from pprint import pprint

import torch
from datasets import load_dataset
from util import preprocess_data, get_output_paths
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


if __name__ == "__main__":

    # First we set up some command line arguments to allow us to specify data/output paths
    # and the number of worker processes without changing the code.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-imdb-model",
        help="A name for the trained model under. A subdirectory with the given name will be created under the `output-path`.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="The root directory under which model checkpoints are stored.",
    )
    parser.add_argument(
        "--logging-path",
        type=str,
        help="The root directory under which logging data (for tensorboard) are stored.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="The number of CPU worker processes to use.",
    )
    args, _ = parser.parse_known_args()

    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    
    # Then we determine the device on which to train the model.
    print("Using PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print(
            f"Rank {rank} of {world_size} (local: {local_rank}) sees {torch.cuda.device_count()} devices"
        )
        device = torch.device("cuda", local_rank)
        print("Using GPU, device name:", torch.cuda.get_device_name(device))
    else:
        print("No GPU found, using CPU instead.")
        device = torch.device("cpu")

    # We also ensure that output paths exist
    output_dir, logging_dir = get_output_paths(args)

    # #### Loading the GPT-neo model
    #
    # We'll use the gpt-neo-1.3B model from the Hugging Face library:
    # https://huggingface.co/EleutherAI/gpt-neo-1.3B
    # Let's start with getting the appropriate tokenizer.
    pretrained_model = "EleutherAI/gpt-neo-1.3B"

    if rank == 0:
        print("Loading model and tokenizer")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the actual base model from Hugging Face
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    model.to(device)
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    # #### Loading the IMDb data set
    #
    # Next we'll load the IMDb data set: https://huggingface.co/docs/datasets/index.
    #
    # The data set contains 100,000 movies reviews from the Internet Movie
    # Database, split into 25,000 reviews for training and 25,000 reviews
    # for testing and 50,000 without labels (unsupervised) that we also use for training.

    train_dataset = load_dataset(
        "imdb", split="train+unsupervised", trust_remote_code=False, keep_in_memory=True
    )
    eval_dataset = load_dataset(
        "imdb", split="test", trust_remote_code=False, keep_in_memory=True
    )

    # Let's print one sample from the dataset.
    if rank == 0:
        print("Sample from dataset")
        pprint(train_dataset[200])

    # #### Setting up the training configuration
    train_batch_size = 32  # This just about fits into the VRAM of a single MI250x GCD with 16-bit floats
    eval_batch_size = 128  # No optimizer state during evaluation, so can use bigger batches for increased throughput

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        logging_dir=logging_dir,
        eval_strategy="steps",
        eval_steps=200,  # compute validation loss every 200 steps
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_train_batch_size=train_batch_size // world_size,
        per_device_eval_batch_size=eval_batch_size,
        max_steps=1000,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],  # log statistics for tensorboard
        ddp_find_unused_parameters=False,  # there are no unused parameters, causing PyTorch to issue a warning should this be set to True
    )

    # #### Preprocessing of training data
    # We tokenize the data into torch tensors, split training into training and validation and set up a collator that
    # is able to arrange single data samples into batches.

    train_dataset_tokenized, validate_dataset_tokenized, eval_dataset_tokenized = preprocess_data(train_dataset, eval_dataset, tokenizer, training_args)

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    # Sanity check: How does the training data look like after preprocessing?
    if rank == 0:
        print("Sample of tokenized data")
        for b in train_dataset_tokenized:
            pprint(b, compact=True)
            print("Length of input_ids:", len(b["input_ids"]))
            break
        print("Length of dataset (tokenized)", len(train_dataset_tokenized))

    # #### Training
    # We use the Hugging Face trainer instead of a manual training loop.
    #
    # You can read about the many, many different parameters to the
    # Hugging Face trainer here:
    # https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
    #

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tokenized,
        eval_dataset=validate_dataset_tokenized,
    )

    # With 1000 steps, batch size 32 and a single GCD, this should take just under 30 minutes.
    trainer.train()

    if rank == 0:
        print()
        print("Training done, you can find all the model checkpoints in", output_dir)

    # #### Evaluating the finetuned model
    with torch.no_grad():
        model.eval()
        # Calculate perplexity
        eval_results = trainer.evaluate()
        test_results = trainer.evaluate(eval_dataset_tokenized)

        if rank == 0:
            print(
                f'Perplexity on validation: {math.exp(eval_results["eval_loss"]):.2f}'
            )
            print(f'Perplexity on test: {math.exp(test_results["eval_loss"]):.2f}')

            # Let's print a few sample generated reviews; this is the same as in the previous exercise
            # but now we use the finetuned model
            prompt = (
                "The movie 'How to run ML on LUMI - A documentation' was great because"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs, do_sample=True, max_length=80, num_return_sequences=4
            )
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print("Sample generated review:")
            for txt in decoded_outputs:
                print("-", txt)
