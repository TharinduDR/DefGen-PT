import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config.model_args import T5Args
from t5.t5_model import T5Model

model_name = "google/mt5-large"
model_type = "mt5"


SEED = 777
full = pd.read_csv("data/pt.defmod.wholeset.tag.tsv", sep="\t")
full["prefix"] = ""
full = full.rename(columns={'context': 'input_text', 'gloss': 'output_text'})

full_train, test = train_test_split(full, test_size=0.2, random_state=SEED)

model_args = T5Args()
model_args.num_train_epochs = 5
model_args.no_save = False
model_args.fp16 = False
model_args.learning_rate = 1e-5
model_args.train_batch_size = 32
model_args.max_seq_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 800
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False
model_args.overwrite_output_dir = True
model_args.save_recent_only = True
model_args.logging_steps = 800
model_args.manual_seed = SEED
model_args.early_stopping_patience = 25
model_args.save_steps = 800

model_args.output_dir = os.path.join("outputs", "mt5")
model_args.best_model_dir = os.path.join("outputs", "mt5", "best_model")
model_args.cache_dir = os.path.join("cache_dir", "mt5")

model_args.wandb_project = "DORE"
model_args.wandb_kwargs = {"name": model_name}

model = T5Model(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available())

train, eval = train_test_split(full_train, test_size=0.2, random_state=SEED)
model.train_model(train, eval_data=eval)

