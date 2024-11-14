import os
from transformers import PreTrainedTokenizerFast, TrainingArguments


def preprocess_data(train_dataset, eval_dataset, tokenizer: PreTrainedTokenizerFast, training_args: TrainingArguments):
    """ Transforms the labelled IMDb data into tokenized tensors for LLM training; further splits train_dataset into training and validation sets.

    Arguments:
        - train_dataset: IMDb training data set split, as loaded by load_dataset.
        - eval_dataset: IMDb testing data set split, as loaded by load_dataset.
        - tokenizer: The tokenizer used with the model to be trained.
        - training_args: The TrainingArguments used for training, to get batch_size and number of workers.
    Returns:
        tuple (train_dataset_tokenized, validate_dataset_tokenized, eval_dataset_tokenized) where
        - train_dataset_tokenized and validate_dataset_tokenized are the tokenized version of train_dataset with an additional subdivision,
        - eval_dataset_tokenized is the tokenized version of eval_dataset.
    """

    # IMDb examples are presented as a dictionary:
    # {
    #    'text': the review text as a string,
    #    'label': a sentiment label as an integer,
    # }.
    # We tokenize the text and add the special token for indicating the end of the
    # text at the end of each review. We also truncate reviews to a maximum
    # length to avoid excessively long sequences during training.
    # As we have no use for the label, we discard it.
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

    train_dataset_tokenized = train_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=training_args.dataloader_num_workers,
    )

    eval_dataset_tokenized = eval_dataset.map(
        tokenize,
        remove_columns=["text", "label"],
        batched=True,
        num_proc=training_args.dataloader_num_workers,
    )

    # We split a small amount of training data as "validation" test set to keep track of evaluation
    # of the loss on non-training data during training.
    # This is purely because computing the loss on the full evaluation dataset takes much longer.
    train_validate_splits = train_dataset_tokenized.train_test_split(
        test_size=1000, seed=42, keep_in_memory=True
    )
    train_dataset_tokenized = train_validate_splits["train"]
    validate_dataset_tokenized = train_validate_splits["test"]

    return train_dataset_tokenized, validate_dataset_tokenized, eval_dataset_tokenized

def get_output_paths(args: argparse.Namespace):
    """ Creates the final output and logging paths from command line arguments and creates the folders, if needed.

    Arguments:
        - args: Namespace object of parsed command line arguments as returned by argparse.ArgumentParser().parse_args()

    Returns:
        tuple (output_dir, logging_dir) where
        - output_dir: the path of the directory in which model checkpoints are to be stored,
        - logging_dir: the path of the directory in which tensorboard logging data are to be stored.
    """
    # this is where trained model and checkpoints will go
    output_dir = os.path.join(args.output_path, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # this is where tensorboard logging outputs will go
    logging_dir = os.path.join(args.logging_path, args.model_name)
    os.makedirs(logging_dir, exist_ok=True)

    return output_dir, logging_dir

