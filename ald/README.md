# Autoregressive Latent Diffusion (ALD) for HELM Peptide Generation

## Architecture Overview

This project implements an **Autoregressive Latent Diffusion** model for generating HELM peptide sequences. Unlike the previous "Global Diffusion" approach (generating the entire sequence at once), ALD generates peptides **token-by-token**, where each token is produced through a complete diffusion denoising process.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Autoregressive Generation Loop                        │
│                                                                          │
│   t=0          t=1          t=2                    t=N-1                 │
│    │            │            │                       │                   │
│    ▼            ▼            ▼                       ▼                   │
│ ┌──────┐    ┌──────┐    ┌──────┐               ┌──────┐                 │
│ │ x_0  │───▶│ x_1  │───▶│ x_2  │───▶  ...  ───▶│ x_N  │                 │
│ └──────┘    └──────┘    └──────┘               └──────┘                 │
│     │            │            │                                          │
│     └────────────┴────────────┴─────── History ─────────┐               │
│                                                          │               │
│                          ┌───────────────────────────────┘               │
│                          ▼                                               │
│                  ┌───────────────┐                                       │
│                  │    Context    │                                       │
│                  │    Encoder    │  ◄── Causal Transformer               │
│                  │   (Brain)     │                                       │
│                  └───────┬───────┘                                       │
│                          │ h_t (context vector)                          │
│                          ▼                                               │
│                  ┌───────────────┐                                       │
│                  │   Diffusion   │  ◄── Complete denoising loop          │
│                  │    Engine     │      z_K → z_{K-1} → ... → z_0        │
│                  │   (Brush)     │                                       │
│                  └───────┬───────┘                                       │
│                          │ clean embedding                               │
│                          ▼                                               │
│                  ┌───────────────┐                                       │
│                  │    Token      │  ◄── Nearest neighbor (cosine sim)    │
│                  │    Mapper     │                                       │
│                  └───────┬───────┘                                       │
│                          │ discrete token x_t                            │
│                          ▼                                               │
│                  ┌───────────────┐                                       │
│                  │  Ring Bond    │  ◄── Predict x_t connects to x_{0..t-1}│
│                  │  Predictor    │                                       │
│                  └───────────────┘                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ald/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration dataclasses
├── core/                       # Core building blocks
│   ├── attention.py            # MultiHeadAttention, CausalAttention, CrossAttention
│   ├── embeddings.py           # Positional encoding, Time embedding, UniMol loader
│   └── layers.py               # FeedForward, TransformerLayer, DenoiserBlock
├── diffusion/                  # Diffusion components
│   ├── schedules.py            # Linear, Cosine variance schedules
│   ├── denoiser.py             # Denoising network architecture
│   └── engine.py               # Forward/reverse diffusion, DDPM/DDIM sampling
├── models/                     # High-level models
│   ├── context_encoder.py      # Causal Transformer for history encoding
│   ├── token_mapper.py         # Embedding → discrete token mapping
│   ├── ring_predictor.py       # Ring bond prediction
│   └── ald_model.py            # Main ALD model combining all components
└── utils/                      # Utilities
    ├── topology.py             # HELM parsing and analysis
    └── data.py                 # Dataset and DataLoader

train_ald.py                    # Training script
generate_ald.py                 # Generation script
```

## Key Components

### 1. Context Encoder (The "Brain")

A **Causal Transformer** that processes the history of previously generated tokens `[x_0, ..., x_{t-1}]` and outputs a context vector `h_t` that conditions the diffusion process.

```python
from ald.models import CausalContextEncoder

encoder = CausalContextEncoder(
    embedding_dim=512,
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=256
)

# Get context for next token generation
context = encoder.get_context_for_next_token(history_embeddings)
```

### 2. Diffusion Engine (The "Brush")

At each autoregressive step, performs a **complete diffusion process** to generate one token embedding:

```python
from ald.diffusion import DiffusionEngine

engine = DiffusionEngine(
    embedding_dim=512,
    num_diffusion_steps=100,
    variance_schedule='cosine'
)

# Sample a single token embedding conditioned on context
embedding = engine.sample(batch_size=1, context=h_t)
```

**Key formulas preserved from original code:**

- Forward process: $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$
- DDPM sampling: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta) + \sigma_t z$

### 3. Token Mapper

Maps continuous embeddings to discrete HELM monomers using **cosine similarity**:

```python
from ald.models import TokenMapper

mapper = TokenMapper(
    vocab=vocab,
    embeddings_dir='./unimol_embeddings',
    use_embedding_norm=True,
    use_freq_weight=True
)

# Map embedding to token (with position constraints)
token_id = mapper(embedding, position=t, seq_len=target_length)
```

### 4. Ring Bond Predictor

Predicts cyclic connections between residues as tokens are generated:

```python
from ald.models import AutoregressiveRingPredictor

predictor = AutoregressiveRingPredictor(d_model=512)

# Predict if current token connects to any previous token
bond = predictor.predict_connection(current_context, history_contexts)
```

## Usage

### Training

```bash
# Basic training
python train_ald.py --data ./data/helm_sequences_chembl32.txt --epochs 100

# With custom configuration
python train_ald.py \
    --data ./data/helm_sequences_chembl32.txt \
    --d_model 512 \
    --context_layers 6 \
    --denoiser_layers 4 \
    --diffusion_steps 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 100 \
    --amp  # Enable mixed precision
```

### Generation

```bash
# Generate with DDPM (slower, higher quality)
python generate_ald.py \
    --checkpoint ./checkpoints/ald/final_model.pt \
    --num_samples 100 \
    --output generated_samples.txt

# Generate with DDIM (faster)
python generate_ald.py \
    --checkpoint ./checkpoints/ald/final_model.pt \
    --num_samples 100 \
    --ddim --ddim_steps 50 \
    --output generated_samples.txt
```

### Python API

```python
import torch
import json
from ald import AutoregressiveLatentDiffusion

# Load vocabulary
with open('./data/helm_vocab.json', 'r') as f:
    vocab = json.load(f)

# Create model
model = AutoregressiveLatentDiffusion(
    vocab=vocab,
    d_model=512,
    n_heads=8,
    context_layers=6,
    denoiser_layers=4,
    num_diffusion_steps=100,
    embeddings_dir='./unimol_embeddings'
)

# Load weights
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Generate sequences
helm_sequences = model.generate_helm_sequences(
    num_samples=10,
    max_length=45,
    min_length=5,
    use_ddim=True,
    ddim_steps=50,
    predict_ring_bonds=True
)

for helm in helm_sequences:
    print(helm)
```

## Mathematical Formulation

### Autoregressive Generation

At each step $t$:

1. **Context Encoding:**
   $$h_t = \text{ContextEncoder}([x_0, x_1, ..., x_{t-1}])$$

2. **Diffusion Sampling:**
   - Start: $z_K \sim \mathcal{N}(0, I)$
   - Denoise: $z_{k-1} = f_\theta(z_k, k, h_t)$ for $k = K, K-1, ..., 1$
   - Output: $\hat{e}_t = z_0$

3. **Token Mapping:**
   $$x_t = \arg\min_{i \in \mathcal{V}_{\text{pos}}} \|normalize(\hat{e}_t) - normalize(E_i)\|$$

   where $\mathcal{V}_{\text{pos}}$ is the set of valid tokens for position $t$.

### Diffusion Process

**Variance Schedule (Cosine):**
$$\bar\alpha_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

**Forward Process:**
$$q(z_k | z_0) = \mathcal{N}(z_k; \sqrt{\bar\alpha_k} z_0, (1-\bar\alpha_k)I)$$

**Reverse Process (DDPM):**
$$p_\theta(z_{k-1}|z_k, h_t) = \mathcal{N}\left(z_{k-1}; \frac{1}{\sqrt{\alpha_k}}\left(z_k - \frac{1-\alpha_k}{\sqrt{1-\bar\alpha_k}}\epsilon_\theta(z_k, k, h_t)\right), \sigma_k^2 I\right)$$

## Configuration

See `ald/config.py` for all configuration options:

```python
from ald.config import ALDConfig, get_default_config

# Default configuration
config = get_default_config()

# Modify as needed
config.model.d_model = 768
config.model.context_layers = 8
config.training.batch_size = 64

# Save/load configuration
config.save('my_config.json')
config = ALDConfig.load('my_config.json')
```

## Differences from Original Global Diffusion

| Aspect | Global Diffusion | Autoregressive Latent Diffusion |
|--------|------------------|----------------------------------|
| Generation | Entire sequence at once | Token-by-token |
| Context | Global (bidirectional) | Causal (previous tokens only) |
| Diffusion | One process per sequence | One process per token |
| Length | Fixed during generation | Variable (dynamic stopping) |
| Ring bonds | Post-hoc prediction | Step-by-step prediction |
| Complexity | $O(T \cdot L^2)$ | $O(L \cdot T)$ where $L$ = length, $T$ = diffusion steps |

## Requirements

- PyTorch >= 1.12
- NumPy
- Pandas (for monomer classification)

## Citation

If you use this code, please cite the relevant works on diffusion models and peptide generation.
