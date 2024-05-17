import argparse
from functools import partial
import math
import multiprocessing
from pathlib import Path
from pprint import pprint
import os

import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from env_utils import print_slurm_env


def tokenize_imdb(dataset, *, tokenizer):
    texts = [example + tokenizer.eos_token for example in dataset["text"]]
    tokens = tokenizer(
        texts,
        max_length=256,
        truncation=True,
        add_special_tokens=True,
        return_overflowing_tokens=True,
        return_length=False,
    )
    return tokens


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=Path,
        help="The root directory under which model checkpoints are stored.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-imdb-model",
        help="A name to store the trained model under.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of CPU worker processes to use.",
    )
    args = parser.parse_args()

    # Setup distributed environment
    multiprocessing.set_start_method(
        # Workaround "fork" not being safe with Slingshot 11 when using multiple
        # PyTorch DataLoader workers
        "spawn"
    )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
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
    os.sched_setaffinity(
        # Set CPU bindings based on LOCAL_RANK which is also used to set GPU device by accelerate
        0,
        LUMI_GPU_CPU_map[local_rank],  # Set CPU binding for the current process (0)
    )
    print_slurm_env()  # Print SLURM environment

    # Setup training configuration
    args.datadir.mkdir(parents=True, exist_ok=True)
    output_dir = args.datadir / args.model_name
    logging_dir = args.datadir / "runs" / args.model_name
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
        ddp_timeout=600,
        disable_tqdm=True,
    )
    pretrained_model = "EleutherAI/gpt-neo-1.3B"

    # Load the IMDB data set
    # Next we'll load the IMDB data set, this time using the Hugging Face
    # datasets library: https://huggingface.co/docs/datasets/index.
    #
    # The dataset contains 100,000 movies reviews from the Internet Movie
    # Database, split into 25,000 reviews for training and 25,000 reviews
    # for testing and 50,000 without labels (unsupervised).
    datasets.disable_progress_bars()
    train_dataset = datasets.load_dataset(
        "imdb", split="train+unsupervised", trust_remote_code=False, keep_in_memory=True
    )
    eval_dataset = datasets.load_dataset(
        "imdb", split="test", trust_remote_code=False, keep_in_memory=True
    )

    # Let's print one sample from the dataset.
    if rank == 0:
        print("Sample from dataset")
        pprint(next(iter(train_dataset)))

    # Tokenize the data
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset_tok = train_dataset.map(
        partial(tokenize_imdb, tokenizer=tokenizer),
        remove_columns=["text", "label"],
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=training_args.dataloader_num_workers,
    )

    eval_dataset_tok = eval_dataset.map(
        partial(tokenize_imdb, tokenizer=tokenizer),
        remove_columns=["text", "label"],
        batched=True,
        num_proc=training_args.dataloader_num_workers,
    )

    # We split a small amount of data as "validation" test set to keep track of
    # evolution of the loss on non-training data during training this is
    # purely because computing the loss on the full evaluation dataset takes
    # much longer
    train_validate_splits = train_dataset_tok.train_test_split(
        test_size=1000, seed=42, keep_in_memory=True
    )
    train_dataset_tok = train_validate_splits["train"]
    validate_dataset_tok = train_validate_splits["test"]

    if rank == 0:
        print("Sample of tokenized data")
        tok_sample = next(iter(train_dataset_tok))
        pprint(tok_sample, compact=True)
        print("Length of input_ids:", len(tok_sample["input_ids"]))
        print("Length of dataset (tokenized)", len(train_dataset_tok))

        num_batches = len(train_dataset) // train_batch_size
        print(f"Training set is comprised of {num_batches} batches")

    # Train the model
    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
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
        print("\nTraining done, you can find all the model checkpoints in", output_dir)

    with torch.no_grad():
        # Calculate perplexity
        eval_results = trainer.evaluate()
        test_results = trainer.evaluate(eval_dataset_tok)

        if rank == 0:
            print(
                f'Perplexity on validation: {math.exp(eval_results["eval_loss"]):.2f}'
            )
            print(f'Perplexity on test: {math.exp(test_results["eval_loss"]):.2f}')

            # Let's print a few sample generated reviews
            prompt = (
                "The movie 'How to run ML on LUMI - A documentation' was great because"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(
                trainer.accelerator.device
            )
            outputs = model.generate(
                **inputs, do_sample=True, max_length=80, num_return_sequences=4
            )
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print("Sample generated review:")
            for txt in decoded_outputs:
                print("-", txt)


if __name__ == "__main__":
    main()
