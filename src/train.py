import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset , random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path


def get_all_sentences(ds, lang):
	for item in ds:
		yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
	tokenizer_path = Path(config["tokenizer_file"].format(lang))
	# use instance method exists() – Path.exists(...) is not correct
	if not tokenizer_path.exists():
		tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
		tokenizer.pre_tokenizer = Whitespace()
		trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
		tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
		tokenizer.save(str(tokenizer_path))
	else:
		tokenizer = Tokenizer.from_file(str(tokenizer_path))
	return tokenizer


def get_ds(config):
	ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}", split="train")

	# Build tokenizers
	tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
	tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

	# Keep 90% for training and 10% for validation
	data_len = len(ds_raw)
	train_size = int(0.9 * data_len)
	val_size = data_len - train_size
	train_ds, val_ds = random_split(ds_raw, [train_size, val_size])

	return train_ds, val_ds, tokenizer_src, tokenizer_tgt
    