# Pico: Tiny Language Models for Learning Dynamics Research

> 🚧 **Coming Soon!** Our complete suite of pre-trained models (1M to 1B parameters) is currently being trained and will be released on [HuggingFace organization](https://huggingface.co/pico-lm) in January 2025.

Pico is a framework designed to facilitate research into language model learning dynamics through a comprehensive suite of small to medium-scale models (1M-1B parameters). Built on a LLAMA-style architecture, Pico emphasizes simplicity, modularity, and research accessibility.

The framework serves two key purposes:
1. **Pre-trained Model Suite**: Access our complete suite of models trained on 420B tokens
2. **Training Framework**: Easily train your own model suite from scratch with minimal setup

This dual-purpose design means researchers can either:
- Use our pre-trained models and checkpoints for immediate analysis
- Train their own suite of models to test specific hypotheses or explore different architectures

## 🔄 Training Philosophy

All models in a Pico suite (whether our pre-trained ones or your custom trained ones):
- Share identical architectures and optimizers
- Train on the same tokens in identical order
- Save rich checkpoint data including activations and gradients
- Enable direct comparisons across model scales

## 📦 Resources

All our pre-trained models and datasets are publicly available through our [HuggingFace organization](https://huggingface.co/pico-lm):
- Pre-trained models (1M to 1B parameters)
- Pre-tokenized training data derived from the DOLMA corpus
- Training checkpoints with activation and gradient information
- Basic evaluation (perplexity) metrics logged throughout training

## 🌟 Why Pico?

Unlike other model suites, Pico is specifically designed for learning dynamics research:

1. **Focused Scale Range**: Covers the critical 1M-1B parameter range where most learning dynamics research is feasible
2. **Consistent Training**: All models see identical data in identical order, enabling true cross-scale comparisons
3. **Rich Analytics**: Automatic saving of activations and gradients for mechanistic interpretability
4. **Research Ready**: Minimal, well-documented code designed to be forked and modified
5. **Clean Data**: Uses a curated, pre-shuffled version of the DOLMA corpus
6. **Train Your Own**: Simple pipeline for training your own suite of models with custom configurations

## 🔑 Key Features

- **Simple Architecture**: Clean, modular implementation of core transformer components
- **Educational Focus**: Well-documented code with clear references to academic papers
- **Research Ready**: Built-in tools for storing and studying model learning dynamics
- **Efficient Training**: Pre-tokenized dataset and optimized training loop
- **Modern Stack**: Built with PyTorch Lightning, Wandb, and HuggingFace integrations

## 🏗️ Core Components

- **RMSNorm** for stable layer normalization
- **Rotary Positional Embeddings (RoPE)** for position encoding
- **Multi-head attention** with KV-cache support
- **SwiGLU activation** function
- **Residual connections** throughout

## 🚀 Quick Start

1. **Clone Project**
```bash
git clone https://github.com/rdiehlmartinez/pico.git && cd pico
```

2. **Configure Environment**
Create `.env` file:
```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
```

3. **Setup Dependencies**
```bash
source setup.sh
```
### Exploring the Codebase

The core implementation is organized into these key files and packages:

- **`src/model/pico.py`**: The heart of Pico
  - LLAMA-style transformer implementation
  - Attention mechanism with KV-cache
  - RoPE positional embeddings
  - Documentation references for each component

- **`src/training/trainer.py`**: Training pipeline
  - Distributed training setup
  - Checkpoint management
  - Logging configuration

- **`src/config`**: Model configuration
  - Hyperparameter definitions
  - Model architecture settings
  - Training parameters

- **`src/checkpointing`**: Checkpointing and State Management
  - Training state persistence (model, optimizer, scheduler)
  - Learning dynamics tracking (activations, weights, gradients)
  - Evaluation results storage
  - Automatically store huggingface-compatible version of model for down-stream use 

### Common Starting Points

1. **Using Pre-trained Models**
```python
from transformers import AutoModelForCausalLM

# Load a specific model size
model = AutoModelForCausalLM.from_pretrained("pico-lm/[...]")
```

2. **Training Your Own Suite**
```bash
# Create a config yaml file, e.g. `my_config.yaml`
# You can follow the provided demo template in configs/demo.yaml
# If no config file is provided the default config values are used
poetry run train --config_path my_config.yaml
```


## 📊 Coming Soon: Pico Analysis

A companion framework for analyzing Pico checkpoints:
- Mechanistic interpretability tools
- Learning dynamics visualization
- Cross-scale model comparisons
- Training trajectory analysis

## 📚 References

Our implementation draws inspiration from and builds upon:
- [LLAMA](https://arxiv.org/abs/2302.13971)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)

## 🤝 Contributing

We welcome contributions in:
- New features and improvements
- Documentation and tutorials
- Bug fixes and testing
- Research findings and analysis


## 📝 License

Apache 2.0 License

## 📫 Contact

- GitHub: [rdiehlmartinez/pico](https://github.com/rdiehlmartinez/pico)
- Author: [Richard Diehl Martinez](https://richarddiehlmartinez.com)

## Citation

If you use Pico in your research, please cite:

```bibtex
@software{pico2024,
    author = {Diehl Martinez, Richard},
    title = {Pico: Framework for Training Tiny Language Models},
    year = {2024},
}
```