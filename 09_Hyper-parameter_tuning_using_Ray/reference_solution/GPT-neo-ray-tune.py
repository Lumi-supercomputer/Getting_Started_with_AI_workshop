#!/usr/bin/env python
import torch
import os
import math
import argparse

from pprint import pprint
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

import ray
from ray import tune, train


def model_training(config):

    args = config["args"]
    learning_rate = config["learning_rate"]

    device = torch.device("cuda")

    # Append the trial ID to the output and logging directories to avoid race conditions
    trial_id = tune.Trainable().trial_id
    output_dir = os.path.join(args.output_path, args.model_name, f"trial_{trial_id}")
    os.makedirs(output_dir, exist_ok=True)

    logging_dir = os.path.join(args.logging_path, args.model_name, f"trial_{trial_id}")
    os.makedirs(logging_dir, exist_ok=True)

    train_dataset = load_dataset(
        "imdb", split="train+unsupervised", trust_remote_code=False, keep_in_memory=True
    )
    eval_dataset = load_dataset(
        "imdb", split="test", trust_remote_code=False, keep_in_memory=True
    )

    pretrained_model = "EleutherAI/gpt-neo-1.3B"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = tokenizer.special_tokens_map

    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    model.to(device)

    train_batch_size = 32
    eval_batch_size = 128

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=not args.resume,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=4,
        logging_dir=logging_dir,
        evaluation_strategy="steps",
        eval_steps=200,
        learning_rate=learning_rate,
        weight_decay=0.01,
        bf16=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        max_steps=100,  # reduced number of steps to keep runtime under 10 min
        dataloader_num_workers=args.num_workers,  # NOTE: setting this causes a crash with LUST EasyBuild PyTorch on multinode. For that software, comment this (but then set num_procs for the data mappings below)
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
    )

    max_length = 256

    def tokenize(x):
        texts = [example + tokenizer.eos_token for example in x["text"]]
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            return_length=False,
        )

    train_dataset_tok = train_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=training_args.dataloader_num_workers,
    )

    eval_dataset_tok = eval_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        num_proc=training_args.dataloader_num_workers,
    )

    train_validate_splits = train_dataset_tok.train_test_split(
        test_size=1000, seed=42, keep_in_memory=True
    )
    train_dataset_tok = train_validate_splits["train"]
    validate_dataset_tok = train_validate_splits["test"]

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset_tok,
        eval_dataset=validate_dataset_tok,
    )

    trainer.train(resume_from_checkpoint=args.resume)

    # report results back to ray
    eval_results = trainer.evaluate()
    train.report(
        dict(
            loss=eval_results["eval_loss"],
            perplexity=math.exp(eval_results["eval_loss"]),
        )
    )


if __name__ == "__main__":
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
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="If set, continue from a previously interrupted run. Otherwise, overwrite existing checkpoints.",
    )
    args, _ = parser.parse_known_args()

    # We need to manually set the number of CPUs and GPUs. Othewise, ray tries to use the whole node and crashes.
    ray.init(num_cpus=56, num_gpus=8, log_to_driver=False)

    config = {
        "learning_rate": tune.uniform(
            1e-6, 1e-3
        ),  # define search space for learning_rate
        "args": args,
    }

    # analysis = tune.run(
    #     model_training,
    #     resources_per_trial={"cpu": 7, "gpu": 1},  # set resources for every trial run
    #     config=config,
    #     num_samples=8,
    #     metric="perplexity",
    #     mode="min",
    # )
    # Define the search algorithm (optional, for more advanced tuning)
    search_alg = tune.search.BasicVariantGenerator()

    # Define the scheduler (optional, for more advanced scheduling)
    scheduler = tune.schedulers.FIFOScheduler()

    # Create a Tuner object
    tuner = Tuner(
        tune.with_resources(
            model_training, resources={"cpu": 7, "gpu": 1}  # Set resources for every trial run
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=8,  # Number of samples
            metric="perplexity",  # Metric to optimize
            mode="min",  # Minimize the metric
        ),
        run_config=ray.tune.RunConfig(
            name="tune_model_training",  # Name of the experiment
            local_dir="./ray_results/",  # Directory to save training results
            stop=None,  # Stopping criteria
        ),
        search_alg=search_alg,
        scheduler=scheduler
    )

    # Run the tuning process
    results = tuner.fit()
    print("Best hyperparameters found were: ", analysis.best_config)
