#!/usr/bin/env python
# coding: utf-8

# # IMDB movie review text generation
#
# In this script, we'll fine-tune a GPT2-like model to generate more
# movie reviews based on a prompt.
#
# Partly based on this tutorial:
# https://github.com/omidiu/GPT-2-Fine-Tuning/


# #### Prelude / Setup
import torch
import os
import psutil
import math
import argparse
import time

from pprint import pprint
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)

def set_cpu_affinity(local_rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)



if __name__ == '__main__':

    # First we set up some command line arguments to allow us to specify data/output paths
    # and the number of worker processes without changing the code.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt-imdb-model", help="A name for the trained model under. A subdirectory with the given name will be created under the `output-path`.")
    parser.add_argument("--output-path", type=str, help="The root directory under which model checkpoints are stored.")
    parser.add_argument("--logging-path", type=str, help="The root directory under which logging data (for tensorboard) are stored.")
    parser.add_argument("--num-workers", type=int, default=1, help="The number of CPU worker processes to use.")
    parser.add_argument("--set-cpu-binds", type=bool, default=False, action="store_true", help="A list of bitmasks (represented as an integer) of the CPUs to which to bind each local process rank. Optional, but if set must provide a mask for each local rank.")
    args, _ = parser.parse_known_args()

    # Read the environment variables provided by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # Set up CPU binding if --set-cpu-binds is given
    if args.set_cpu_binds:
        set_cpu_affinity(local_rank)

    # Then we determine the device on which to train the model.
    print('Using PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print(f"Rank {rank} of {world_size} (local: {local_rank}) sees {torch.cuda.device_count()} devices")
        print('Using GPU, device name:', torch.cuda.get_device_name(local_rank))
        device = torch.device('cuda', local_rank)
    else:
        print('No GPU found, using CPU instead.')
        device = torch.device('cpu')

    # We also ensure that output paths exist
    output_dir = os.path.join(args.output_path, args.model_name)  # this is where trained model and checkpoints will go
    os.makedirs(output_dir, exist_ok=True)

    logging_dir = os.path.join(args.logging_path, args.model_name)  # this is where tensorboard logging outputs will go
    os.makedirs(logging_dir, exist_ok=True)

    # #### Loading the IMDb data set
    #
    # Next we'll load the IMDb data set: https://huggingface.co/docs/datasets/index.
    #
    # The data set contains 100,000 movies reviews from the Internet Movie
    # Database, split into 25,000 reviews for training and 25,000 reviews
    # for testing and 50,000 without labels (unsupervised) that we also use for training.

    train_dataset = load_dataset("imdb", split="train+unsupervised", trust_remote_code=False, keep_in_memory=True)
    eval_dataset = load_dataset("imdb", split="test", trust_remote_code=False, keep_in_memory=True)

    # Let's print one sample from the dataset.
    if rank == 0:
        print('Sample from dataset')
        for b in train_dataset:
            pprint(b)
            break

    # #### Loading the GPT-neo model
    #
    # We'll use the gpt-neo-1.3B model from the Hugging Face library:
    # https://huggingface.co/EleutherAI/gpt-neo-1.3B
    # Let's start with getting the appropriate tokenizer.
    pretrained_model = "EleutherAI/gpt-neo-1.3B"

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map

    # Load the actual base model from Hugging Face
    if rank == 0:
        print("Loading model and tokenizer")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    model.to(device)
    stop = time.time()
    if rank == 0:
        print(f"Loading model and tokenizer took: {stop-start:.2f} seconds")

    # #### Setting up the training configuration
    train_batch_size = 32  # This just about fits into the VRAM of a single MI250x GCD with 16-bit floats
    eval_batch_size = 128  # No optimizer state during evaluation, so can use bigger batches for increased throughput

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        logging_dir=logging_dir,
        evaluation_strategy="steps",
        eval_steps=200,  # compute validation loss every 200 steps
        learning_rate=2e-5,
        weight_decay=0.01,
        bf16=True,  # use 16-bit floating point precision
        per_device_train_batch_size=train_batch_size // world_size,  # divide the total training batch size by the number of GCDs for the per-device batch size
        per_device_eval_batch_size=eval_batch_size,
        max_steps=1000,
        dataloader_num_workers=args.num_workers, # NOTE: setting this causes a crash with LUST EasyBuild PyTorch on multinode. For that software, comment this (but then set num_procs for the data mappings below)
        dataloader_pin_memory=True,
        report_to=["tensorboard"], # log statistics for tensorboard
        ddp_find_unused_parameters=False, # there are no unused parameters, causing PyTorch to issue a warning should this be set to True
    )

    # #### Setting up preprocessing of training data

    # IMDb examples are presented as a dictionary:
    # {
    #    'text': the review text as a string,
    #    'label': a sentiment label as an integer,
    # }.
    # We tokenize the text and add the special token for indicating the end of the
    # text at the end of each review. We also truncate reviews to a maximum
    # length to avoid excessively long sequences during training.
    # As we have no use for the label, we discard it.
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


    # We split a small amount of training data as "validation" test set to keep track of evaluation
    # of the loss on non-training data during training.
    # This is purely because computing the loss on the full evaluation dataset takes much longer.
    train_validate_splits = train_dataset_tok.train_test_split(test_size=1000, seed=42, keep_in_memory=True)
    train_dataset_tok = train_validate_splits['train']
    validate_dataset_tok = train_validate_splits['test']

    # Sanity check: How does the training data look like after preprocessing?
    if rank == 0:
        print('Sample of tokenized data')
        for b in train_dataset_tok:
            pprint(b, compact=True)
            print('Length of input_ids:', len(b['input_ids']))
            break
        print('Length of dataset (tokenized)', len(train_dataset_tok))


    # #### Training
    # We use the Hugging Face trainer instead of a manual training loop.
    # 
    # You can read about the many, many different parameters to the
    # Hugging Face trainer here:
    # https://huggingface.co/docs/transformers/v4.37.0/en/main_classes/trainer#transformers.TrainingArguments
    #

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors='pt')

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tok,
        eval_dataset=validate_dataset_tok,
    )

    # With 1000 steps, batch size 32 and a single GCD, this should take just under 30 minutes.
    trainer.train()

    if rank == 0:
        print()
        print("Training done, you can find all the model checkpoints in", output_dir)

    # #### Evaluating the finetuned model
    with torch.no_grad():
        # Calculate perplexity
        eval_results = trainer.evaluate()
        test_results = trainer.evaluate(eval_dataset_tok)

        if rank == 0:
            print(f'Perplexity on validation: {math.exp(eval_results["eval_loss"]):.2f}')
            print(f'Perplexity on test: {math.exp(test_results["eval_loss"]):.2f}')

            # Let's print a few sample generated reviews; this is the same as in the previous exercise
            # but now we use the finetuned model
            prompt = "The movie 'How to run ML on LUMI - A documentation' was great because"
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model.generate(**inputs, do_sample=True, max_length=80, num_return_sequences=4)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print('Sample generated review:')
            for txt in decoded_outputs:
                print('-', txt)
