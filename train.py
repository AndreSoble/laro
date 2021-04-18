import argparse

import torch
import warnings
from pprint import pprint

from torch.autograd import profiler


from torch.optim import Adam
from tqdm import tqdm
from random import shuffle
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import EvaluationStrategy

from modeling_laro import LARO
from train_utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

if args.local_rank == 0:
    import wandb
    wandb.login()

warnings.filterwarnings("ignore")

assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage"))
corpus = Corpus(downsampled=bool(int(os.environ.get("DOWNSAMPLE", 1))),
                downsampled_count=int(os.environ.get("DOWNSAMPLE_COUNT", 1000)))
corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG", 1))), path=os.environ.get("DATA_DIR", "./storage"))

train_dataset = DataLoader(corpus.get_train(shuffled=True))
test_dataset = DataLoader(corpus.get_dev() + corpus.get_eval())

pprint(corpus.get_data_counts())

model = LARO.from_pretrained('xlm-roberta-base')
training_args = TrainingArguments(
    output_dir=os.environ.get("OUTPUT_DIR", './results'),  # output directory
    num_train_epochs=int(os.environ.get("EPOCHS", 1)),  # total # of training epochs
    per_device_train_batch_size=int(os.environ.get("PER_DEVICE_BATCH_SIZE", 40)),
    per_device_eval_batch_size=int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 40)),  # batch size for evaluation
    warmup_steps=int(os.environ.get("STEPS", 100)),
    save_steps=int(os.environ.get("STEPS", 100)),
    logging_steps=int(os.environ.get("STEPS", 1)),
    logging_dir=os.environ.get("LOG_DIR", './logs'),
    learning_rate=float(os.environ.get("LR", 5e-5)),
    fp16=True,
    evaluation_strategy=EvaluationStrategy.STEPS,
    save_total_limit=3,
    prediction_loss_only=True,
    report_to='wandb' if args.local_rank == 0 else None,  # enable logging to W&B
    run_name=os.environ.get("RUN_NAME", 'laro_training_fast_deepspeed_test') if args.local_rank == 0 else None,  # name of the W&B run (optional),
    sharded_ddp=True,
    local_rank=args.local_rank
)

trainer = CustomTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    data_collator=data_collector
)

output = trainer.train()
if args.local_rank == 0:
    wandb.finish()
