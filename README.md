 # 🎯 Pico: Tiny Language Models for Learning Dynamics Research

Pico is a framework for training and analyzing small language models, designed with clarity and educational purposes in mind. Built on a LLAMA-style architecture, Pico makes it easy to experiment with and understand transformer-based language models.

## 🔑 Key Features

- **Simple Architecture**: Clean, modular implementation of core transformer components
- **Educational Focus**: Well-documented code with clear references to academic papers
- **Research Ready**: Built-in tools for analyzing model learning dynamics
- **Efficient Training**: Pre-tokenized dataset and optimized training loop
- **Modern Stack**: Built with PyTorch Lightning, Wandb, and HuggingFace integrations

## 🏗️ Core Components

- **RMSNorm** for stable layer normalization
- **Rotary Positional Embeddings (RoPE)** for position encoding
- **Multi-head attention** with KV-cache support
- **SwiGLU activation** function
- **Residual connections** throughout

## 📚 References

Our implementation draws inspiration from and builds upon:
- [LLAMA](https://arxiv.org/abs/2302.13971)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)

## 🤝 Contributing

We welcome contributions! Whether it's:
- Adding new features
- Improving documentation
- Fixing bugs
- Sharing experimental results

## 📝 License

Apache 2.0 License

## 📫 Contact

- GitHub: [rdiehlmartinez/pico](https://github.com/rdiehlmartinez/pico)
- Author: Richard Diehl Martinez

## 🔍 Citation

If you use Pico in your research, please cite:

@software{pico2024,
author = {Martinez, Richard Diehl},
title = {Pico: Framework for Training Tiny Language Models},
year = {2024},
}
