# Transformer From Scratch

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a complete implementation of the Transformer architecture described in the paper **"Attention Is All You Need"** by Vaswani et al. (2017). The implementation is built from scratch using PyTorch and focuses on machine translation between English and French.

I built it by following Umar Jamil's walkthrough ["Coding a Transformer from scratch on PyTorch, with full explanation, training and inference"](https://www.youtube.com/watch?v=ISNdQcPhsts) and its reference code, [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer). The reference translates English to Italian; this version targets English to French on OPUS Books.

## Status

The model, tokenizers, training and validation loops, and greedy decoding are implemented. Training was started but not completed, because I had no GPU. There are no trained weights or results.

## Introduction

The Transformer model revolutionized the field of natural language processing by introducing a novel architecture based entirely on attention mechanisms, eliminating the need for recurrent or convolutional layers. This is a learning project: an educational re-implementation of the original Transformer architecture.

### Key Contributions of the Original Paper:
- **Self-Attention Mechanism**: Allows the model to focus on different parts of the input sequence
- **Multi-Head Attention**: Enables the model to jointly attend to information from different representation subspaces
- **Positional Encoding**: Injects sequence order information without using recurrent connections
- **Encoder-Decoder Architecture**: Processes input sequences and generates output sequences efficiently

## Architecture Overview

The implementation follows the original paper's architecture with the following components:

```
Input Embeddings + Positional Encoding
           ↓
    ┌─────────────┐
    │   Encoder   │ (N=6 layers)
    │   - Multi-Head Self-Attention
    │   - Feed Forward Network
    │   - Residual Connections + Layer Norm
    └─────────────┘
           ↓
    ┌─────────────┐
    │   Decoder   │ (N=6 layers)
    │   - Masked Multi-Head Self-Attention
    │   - Multi-Head Cross-Attention
    │   - Feed Forward Network
    │   - Residual Connections + Layer Norm
    └─────────────┘
           ↓
    Linear Projection + Softmax
```

## Features

- ✅ **Complete Transformer Implementation**: All components from the original paper
- ✅ **Bilingual Dataset Support**: English-French translation using OPUS Books dataset
- ✅ **Custom Tokenizer**: Word-level tokenization with special tokens
- ✅ **Training Pipeline**: Complete training loop with validation
- ✅ **TensorBoard Integration**: Real-time training monitoring
- ✅ **Model Checkpointing**: Saves a checkpoint after each epoch and resumes from one via `preload`
- ✅ **Greedy Decoding**: Inference with greedy search algorithm
- ✅ **Configurable Architecture**: Easy parameter modification
- ✅ **Clean Code Structure**: Well-documented and modular design

## Requirements

- Python 3.11+
- PyTorch 2.8+
- CUDA (optional, for GPU acceleration)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Chamiln17/Transformer-From-Scratch.git
cd Transformer-From-Scratch
```

2. **Install dependencies with [uv](https://docs.astral.sh/uv/):**
```bash
uv sync
```

3. **Verify installation:**
```bash
uv run python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Dataset

The implementation uses the **OPUS Books** dataset for English-French translation, which is automatically downloaded and processed during training. The dataset includes:

- **Source Language**: English
- **Target Language**: French
- **Training Split**: 90% of the data
- **Validation Split**: 10% of the data
- **Special Tokens**: `[SOS]`, `[EOS]`, `[PAD]`, `[UNK]`

### Dataset Statistics
- Maximum source sequence length: Variable (configurable)
- Maximum target sequence length: Variable (configurable)
- Vocabulary sizes: Automatically determined from the dataset

## Usage

### Training

To train the model from scratch:

```bash
uv run python src/train.py
```

The training script will:
1. Download and preprocess the OPUS Books dataset
2. Build custom tokenizers for both languages
3. Initialize the Transformer model
4. Start training with TensorBoard logging
5. Save model checkpoints after each epoch

Monitor training with TensorBoard:

```bash
uv run tensorboard --logdir runs/tmodel
```

### Evaluation

During training, the model automatically runs validation at the end of each epoch and displays:
- Source sentence (English)
- Target sentence (French)
- Predicted translation

## Model Architecture

### Hyperparameters (Default Configuration)

| Parameter    | Value | Description                      |
| ------------ | ----- | -------------------------------- |
| `d_model`    | 512   | Model dimension                  |
| `h`          | 8     | Number of attention heads        |
| `d_ff`       | 2048  | Feed-forward network dimension   |
| `N`          | 6     | Number of encoder/decoder layers |
| `dropout`    | 0.1   | Dropout rate                     |
| `seq_len`    | 400   | Maximum sequence length          |
| `batch_size` | 4     | Training batch size              |
| `lr`         | 1e-4  | Learning rate                    |
| `num_epochs` | 10    | Number of training epochs        |

### Key Components

1. **Input Embedding**: Converts token IDs to dense vectors
2. **Positional Encoding**: Adds position information using sine/cosine functions
3. **Multi-Head Attention**: 
   - Self-attention in encoder
   - Masked self-attention in decoder
   - Cross-attention between encoder and decoder
4. **Feed Forward Network**: Two linear layers with ReLU activation
5. **Layer Normalization**: Applied before each sublayer (Pre-LN)
6. **Residual Connections**: Skip connections around each sublayer

## Configuration

Modify the configuration in `src/config.py`:

```python
def get_config():
    return {
        "batch_size": 4,           # Adjust based on GPU memory
        "num_epochs": 10,          # Number of training epochs
        "lr": 10**-4,              # Learning rate
        "seq_len": 400,            # Maximum sequence length
        "d_model": 512,            # Model dimension
        "h": 8,                    # Attention heads
        "d_ff": 2048,              # Feed-forward dimension
        "N": 6,                    # Encoder/decoder layers
        "dropout": 0.1,            # Dropout rate
        "lang_src": "en",          # Source language
        "lang_tgt": "fr",          # Target language
        "model_folder": "weights", # Model save directory
        "model_basename": "tmodel_",
        "preload": None,           # Resume from checkpoint
        "tokenizer_name": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
```

## Project Structure

```
transformer-from-scratch/
├── src/
│   ├── model.py          # Transformer architecture implementation
│   ├── train.py          # Training pipeline
│   ├── dataset.py        # Dataset handling and preprocessing
│   └── config.py         # Configuration management
├── weights/              # Model checkpoints (created during training)
├── runs/                 # TensorBoard logs (created during training)
├── tokenizer_en.json     # English tokenizer (built on first run if missing)
├── tokenizer_fr.json     # French tokenizer (built on first run if missing)
├── pyproject.toml        # Project dependencies
├── README.md            # This file
└── LICENSE              # MIT License
```

### Code Organization

- **`model.py`**: Contains all Transformer components (attention, encoder, decoder, etc.)
- **`train.py`**: Main training script with data loading, training loop, and validation
- **`dataset.py`**: Custom dataset class for bilingual translation data
- **`config.py`**: Centralized configuration management

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Umar Jamil's video ["Coding a Transformer from scratch on PyTorch, with full explanation, training and inference"](https://www.youtube.com/watch?v=ISNdQcPhsts) and reference code [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer), which this project follows. The reference translates English to Italian; this version targets English to French on OPUS Books.
- Original Transformer paper by Vaswani et al.
- PyTorch team for the excellent deep learning framework
- Hugging Face for the datasets library
- The open-source community for inspiration and feedback

---

**Note**: This implementation is designed for educational purposes and to demonstrate the Transformer architecture. For production use, consider using established libraries like Hugging Face Transformers or implementing additional optimizations.