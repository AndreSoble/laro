import torch
import warnings
from torch.optim import Adam
from tqdm import tqdm
from random import shuffle
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import EvaluationStrategy
from modeling_labse import LARO
from train_utils import *

warnings.filterwarnings("ignore")

assert download_and_extract(path=os.environ.get("DATA_DIR", "./storage"))
corpus = Corpus(downsampled=bool(int(os.environ.get("DOWNSAMPLE", 1))),
                downsampled_count=int(os.environ.get("DOWNSAMPLE_COUNT", 10)))
corpus.load_corpus(debug=bool(int(os.environ.get("DEBUG", 1))), path=os.environ.get("DATA_DIR", "./storage"))

train_dataset = DataLoader(corpus.get_train(shuffled=True))
test_dataset = train_dataset#DataLoader(corpus.get_dev())
eval_dataset = train_dataset#DataLoader(corpus.get_eval())

model = LARO.from_pretrained('xlm-roberta-base')
training_args = TrainingArguments(
    output_dir=os.environ.get("OUTPUT_DIR", './results'),  # output directory
    num_train_epochs=int(os.environ.get("EPOCHS", 10)),  # total # of training epochs
    per_device_train_batch_size=int(os.environ.get("PER_DEVICE_BATCH_SIZE", 10)),
    per_device_eval_batch_size=int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 10)),  # batch size for evaluation
    warmup_steps=int(os.environ.get("STEPS", 5)),
    save_steps=int(os.environ.get("STEPS", 5)),
    logging_steps=int(os.environ.get("STEPS", 5)),  # number of warmup steps for learning rate scheduler
    weight_decay=float(os.environ.get("WEIGHT_DECAY", 0.01)),  # strength of weight decay
    logging_dir=os.environ.get("LOG_DIR", './logs'),  # directory for storing logs
    learning_rate=1e-4,
    #fp16=torch.cuda.is_available(),
    evaluation_strategy=EvaluationStrategy.EPOCH,
    save_total_limit=5,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=test_dataset,  # evaluation dataset
    data_collator=data_collector,
)

output = trainer.train()


