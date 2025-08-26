import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset , random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset ,causal_mask
from  model import build_transformer
from config import get_config , get_weights_file_path

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
	train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

	train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
	val_ds = BilingualDataset(val_ds_raw,tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
 
	max_len_src = 0
	max_len_tgt = 0
	for item in ds_raw:
		src_ids=tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
		tgt_ids=tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
		max_len_src= max(max_len_src, len(src_ids) )
		max_len_tgt= max(max_len_tgt,len(tgt_ids))
  
  
	print(f'Max length in source sentence of lang{config["lang_src"]} : {max_len_src}')
	print(f'Max length in target sentence of lang{config["lang_tgt"]} : {max_len_tgt}')
	
	train_dataloader= DataLoader(train_ds,batch_size=config["batch_size"], shuffle=True)
	val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
 
	return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
 
def get_model(config, vocab_size_src:int, vocab_size_tgt:int):
    model = build_transformer(src_vocab_size=vocab_size_src,
        tgt_vocab_size=vocab_size_tgt,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
		d_model=config["d_model"])
    return model
def train_model(config):
    # define the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using device : {device}')
    
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config=config)    
    model = get_model(config=config, vocab_size_src=tokenizer_src.get_vocab_size(), vocab_size_tgt=tokenizer_tgt.get_vocab_size())
    
    # TensorBoard to visualize the summary 
    
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"] , eps=1e-9)
    
    initiale_epoch =0
    global_step=0
    if config["preload"]:
        model_filename=get_weights_file_path(config, config["preload"])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initiale_epoch=state["epoch"] +1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), lable_smoothing=0.1).to(device)