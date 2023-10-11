import os

import torch
from sklearn.model_selection import train_test_split
import pandas as pd

from config.model_args import Seq2SeqArgs
from experiments.evaluate import bleu, ter, bertscore, bleurt_score
from seq2seq.seq2seq_model import Seq2SeqModel

model_name = "facebook/mbart-large-cc25"
model_type = "mbart"

SEED = 777
full = pd.read_csv("data/pt.defmod.wholeset.tag.tsv", sep="\t")
full = full.rename(columns={'context': 'input_text', 'gloss': 'target_text'})

full_train, test = train_test_split(full, test_size=0.2, random_state=SEED)

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.no_save = False
model_args.fp16 = False
model_args.learning_rate = 1e-4
model_args.train_batch_size = 8
model_args.max_seq_length = 256
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = 3200
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False
model_args.overwrite_output_dir = True
model_args.save_recent_only = True
model_args.logging_steps = 3200
model_args.manual_seed = SEED
model_args.early_stopping_patience = 25
model_args.save_steps = 3200

model_args.output_dir = os.path.join("outputs", "mbart50")
model_args.best_model_dir = os.path.join("outputs", "mbart50", "best_model")
model_args.cache_dir = os.path.join("cache_dir", "mbart50")

model_args.wandb_project = "DORE"
model_args.wandb_kwargs = {"name": model_name}

model = Seq2SeqModel(
    encoder_decoder_type=model_type,
    encoder_decoder_name=model_name,
    args=model_args,
    use_cuda=torch.cuda.is_available()
)

train, eval = train_test_split(full_train, test_size=0.2, random_state=SEED)
model.train_model(train, eval_data=eval)

input_list = test['input_text'].tolist()
truth_list = test['target_text'].tolist()

model = Seq2SeqModel(
    encoder_decoder_type=model_args.best_model_dir,
    encoder_decoder_name=model_name,
    args=model_args,
    use_cuda=torch.cuda.is_available()
)

preds = model.predict(input_list)

del model

print("Bleu Score ", bleu(truth_list, preds))
print("Ter Score ", ter(truth_list, preds))
print("BERTScore ", bertscore(truth_list, preds))
print("Bleurt ", bleurt_score(truth_list, preds))
