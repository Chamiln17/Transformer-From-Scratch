from pathlib import Path
def get_config():
    return {
        "batch_size":4,
        "num_epochs":10,
        "lr":10**-4,
        "seq_len":400,
        "d_model":512,
        "h":8,
        "d_ff":2048,
        "N":6,
        "dropout":0.1,
        "lang_src":"en",
        "lang_tgt":"fr",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_name":"tokenizer_{0}.json",
        "experiment_name":"runs/tmodel"
    }
def get_weights_file_path(config, epoch:str):
    model_folder=config["model_folder"]
    model_basename=config["model_basename"]
    model_filename= f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder /model_filename )