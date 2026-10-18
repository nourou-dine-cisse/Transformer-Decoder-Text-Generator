# Autoregressive Decoder-Only Transformer from Scratch (LLaMA / Mistral Architecture)

> **Academic Context**: ENSIIE & Université Paris-Saclay — 3rd Year Engineering / Master 2 Artificial Intelligence (*NLP Today*)  
> **Author**: Mohamed Nourou Dine CISSE  
> **Domain**: `Deep Learning & Natural Language Processing (NLP)`  
> **Core Tech Stack**: `Python` `PyTorch` `Hugging Face Transformers` `Hugging Face Datasets` `RoPE` `SwiGLU` `KV Cache`

---

## 📌 Project Overview (Honest & Grounded)

This repository contains a complete, tensor-level **from-scratch PyTorch implementation** of a modern **decoder-only Transformer** language model inspired by cutting-edge open-weight architectures such as **LLaMA 2** and **Mistral**, implemented without relying on PyTorch's high-level `nn.TransformerDecoder` abstraction.

The primary objective of this project was to implement, mathematically verify, and experimentally evaluate each foundational building block of contemporary Large Language Models (LLMs):
1. **Multi-Head Causal Self-Attention**: Scaled dot-product attention with strict upper-triangular causal masking ($-\infty$).
2. **Rotary Position Embeddings (RoPE)**: Direct rotation of query and key projections to preserve relative token distances without additive positional encodings.
3. **RMSNorm (Root Mean Square Normalization)**: Pre-layer normalization before self-attention and feed-forward sub-blocks.
4. **SwiGLU Feed-Forward Network**: Gated linear unit with SiLU activation, employing Gate, Up, and Down projections.
5. **Key-Value (KV) Caching**: Optimized autoregressive decoding state management, reducing next-token generation complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.
6. **Autoregressive Decoding Strategies**: Efficient greedy decoding and stochastic temperature-scaled sampling.
7. **Rigorous Evaluation**: Cross-entropy loss tracking and perplexity ($\text{PPL} = \exp(\text{loss})$) computation on validation data.

---

## 🏗️ Mathematical Formulation & Architecture

### 1. Multi-Head Causal Attention with Cache Support
Given query, key, and value tensors $Q, K, V \in \mathbb{R}^{B \times H \times L \times D_h}$ and a causal mask $M \in \mathbb{R}^{L_q \times L_k}$ where $M_{ij} = -\infty$ for future token positions:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{D_h}} + M\right) V$$

When decoding token-by-token with KV caching:
- The new token produces $K_{\text{new}}$ and $V_{\text{new}}$.
- Past tensors are concatenated: $K = [K_{\text{cache}} \,\|\, K_{\text{new}}]$ and $V = [V_{\text{cache}} \,\|\, V_{\text{new}}]$.
- Attention is computed only for the active query vector against the entire accumulated sequence.

### 2. Rotary Position Embeddings (RoPE)
Instead of adding positional vectors to token embeddings, RoPE applies a 2D rotational matrix to pairs of coordinates in the projected query and key vectors:

$$R_{\Theta, m}^{2d} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$$

This guarantees that:
$$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = g(q, k, m - n)$$
meaning the dot-product depends strictly on the relative distance $m - n$ between tokens.

### 3. Pre-Norm Decoder Block
Each block incorporates residual skip connections with RMSNorm applied prior to each transformation:

$$x^{(1)} = x + \text{Attention}(\text{RMSNorm}(x))$$
$$x^{(2)} = x^{(1)} + \text{FeedForward}(\text{RMSNorm}(x^{(1)}))$$

$$\text{SwiGLU}(x) = (\text{Linear}_{\text{gate}}(x) \odot \text{SiLU}(\text{Linear}_{\text{up}}(x))) \cdot W_{\text{down}}$$

---

## 📂 Repository Structure

```text
16_Transformer_Decoder_Text_Generator/
├── README.md                      # Comprehensive technical documentation (English)
├── rope_embedding.py              # Self-contained RoPE module with precomputed cos/sin tables
├── implement_transformer.ipynb    # Interactive notebook: unit tests, training loop, generation
└── src/
    ├── __init__.py                # Python package exports
    ├── model.py                   # PyTorch modules: RoPEAttention, TransformerFeedForward, TransformerDecoder
    └── generate.py                # Autoregressive decoding (greedy, temperature sampling, KV-cache, perplexity)
```

---

## ⚙️ Dataset & Model Configuration

- **Training Dataset**: `roneneldan/TinyStories` (Hugging Face Datasets) — synthetic short stories with simple vocabulary, designed for rapid validation and training of small language models.
- **Tokenizer**: `AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")` (or compatible LLaMA-based BPE tokenizer).
- **Default Educational Model Sizing**:
  - Vocabulary Size: ~32,768 tokens
  - Hidden Dimension ($d_{\text{model}}$): 128
  - FFN Intermediate Dimension ($d_{\text{ff}}$): 256
  - Number of Attention Heads ($H$): 4
  - Number of Hidden Blocks: 5 layers
  - Optimizer: AdamW ($\text{lr} = 10^{-3}$)

---

## 🚀 How to Run & Quickstart

### Prerequisites
Install the required Python packages:
```bash
pip install torch transformers datasets tqdm
```

### 1. Interactive Jupyter Notebook
Launch Jupyter to explore step-by-step tensor tests, mask assertions, training diagnostics, and sample completions:
```bash
jupyter notebook implement_transformer.ipynb
```

### 2. Standalone Python Usage
You can directly import and run the modular implementation from Python:
```python
import torch
from transformers import AutoTokenizer
from src.model import TransformerDecoder
from src.generate import sampling_decoding

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

# Initialize custom decoder-only Transformer
model = TransformerDecoder(
    vocabulary_size=len(tokenizer),
    embed_size=128,
    intermediate_size=256,
    num_heads=4,
    hidden_layers=5
)

# Generate continuation with KV caching and temperature sampling
prompt = "Once upon a time, a little girl named Alice"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output_ids = sampling_decoding(model, input_ids, max_new_tokens=48, temperature=0.7)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---

## 🔍 Honest Technical Assessment & Limitations

- **Pedagogical Scope**: This is an academic research & learning implementation created during the *NLP Today* course at ENSIIE. Its purpose is educational: providing clear, transparent PyTorch code detailing the exact tensor algebra behind modern open-weight LLMs.
- **Scale**: The default hyperparameters (128 embedding dimensions, 5 layers) represent a micro-model designed to run and iterate quickly on standard CPU or a single modest GPU. It is not intended to compete with multi-billion-parameter foundation models.
- **KV Cache Benchmarking**: The custom KV-caching implementation eliminates redundant quadratic prefix recomputations during generation, keeping inference latency bounded per token.

---

## 📬 Author Profile
- **Mohamed Nourou Dine CISSE**
- **ENSIIE** & **Université Paris-Saclay** — Engineering Degree & Master 2 Artificial Intelligence (M2IA)
- Completed Academic Exchange: **McMaster University** (Canada, Winter 2026)
- **GitHub**: [@nourou-dine-cisse](https://github.com/nourou-dine-cisse)
- **Email**: [mouhamed.nouroudine.cisse@gmail.com](mailto:mouhamed.nouroudine.cisse@gmail.com)
