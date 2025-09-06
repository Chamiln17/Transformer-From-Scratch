# Transformer From Scratch

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains a complete implementation of the Transformer architecture described in the paper **"Attention Is All You Need"** by Vaswani et al. (2017). The implementation is built from scratch using PyTorch and focuses on machine translation between English and French.

## Table of Contents

- [Transformer From Scratch](#transformer-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Key Contributions of the Original Paper:](#key-contributions-of-the-original-paper)
  - [Architecture Overview](#architecture-overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Dataset](#dataset)
    - [Dataset Statistics](#dataset-statistics)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Inference](#inference)
  - [Model Architecture](#model-architecture)
    - [Hyperparameters (Default Configuration)](#hyperparameters-default-configuration)
    - [Key Components](#key-components)
  - [Configuration](#configuration)
  - [Results](#results)
    - [Training Progress](#training-progress)
    - [Expected Performance](#expected-performance)
    - [Sample Translations](#sample-translations)
  - [Project Structure](#project-structure)
    - [Code Organization](#code-organization)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Contributing](#contributing)
  - [Support](#support)

## Introduction

The Transformer model revolutionized the field of natural language processing by introducing a novel architecture based entirely on attention mechanisms, eliminating the need for recurrent or convolutional layers. This implementation provides a clean, educational, and production-ready version of the original Transformer architecture.

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
- ✅ **Model Checkpointing**: Save and resume training functionality
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
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch
```

2. **Install dependencies using uv (recommended):**
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
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
python src/train.py
```

The training script will:
1. Download and preprocess the OPUS Books dataset
2. Build custom tokenizers for both languages
3. Initialize the Transformer model
4. Start training with TensorBoard logging
5. Save model checkpoints after each epoch

### Evaluation

During training, the model automatically runs validation every few steps and displays:
- Source sentence (English)
- Target sentence (French)
- Predicted translation

### Inference

To use a trained model for inference:

```python
from src.model import build_transformer
from src.config import get_config
import torch

# Load configuration
config = get_config()

# Build model
model = build_transformer(
    src_vocab_size=config["src_vocab_size"],
    tgt_vocab_size=config["tgt_vocab_size"],
    src_seq_len=config["seq_len"],
    tgt_seq_len=config["seq_len"],
    d_model=config["d_model"]
)

# Load trained weights
checkpoint = torch.load("weights/tmodel_epoch09.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Run inference
model.eval()
# ... (inference code)
```

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
        "lang_src": "en",          # Source language
        "lang_tgt": "fr",          # Target language
        "model_folder": "weights", # Model save directory
        "model_basename": "tmodel_",
        "preload": None,           # Resume from checkpoint
        "tokenizer_name": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }
```

## Results

### Training Progress

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir runs/tmodel
```

### Expected Performance

- **Training Loss**: Should decrease steadily over epochs
- **Validation BLEU**: Will improve as the model learns translation patterns
- **Translation Quality**: Improves significantly after several epochs

### Sample Translations

After training, the model can produce translations like:

```
SOURCE: Hello, how are you today?
TARGET: Bonjour, comment allez-vous aujourd'hui ?
PREDICTED: Bonjour, comment allez-vous aujourd'hui ?
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
├── tokenizer_en.json     # English tokenizer (created during training)
├── tokenizer_fr.json     # French tokenizer (created during training)
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

- Original Transformer paper by Vaswani et al.
- PyTorch team for the excellent deep learning framework
- Hugging Face for the datasets library
- The open-source community for inspiration and feedback

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions, please open an issue on GitHub or reach out to the maintainers.

---

**Note**: This implementation is designed for educational purposes and to demonstrate the Transformer architecture. For production use, consider using established libraries like Hugging Face Transformers or implementing additional optimizations.