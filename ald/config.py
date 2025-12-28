"""
Configuration for Autoregressive Latent Diffusion (ALD) model.

Defines hyperparameters and settings for training and generation.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List
from pathlib import Path


@dataclass
class ALDModelConfig:
    """Model architecture configuration."""
    
    # Embedding dimensions
    embedding_dim: int = 512  # Will be updated from Uni-Mol embeddings
    d_model: int = 512
    
    # Attention configuration
    n_heads: int = 8
    
    # Layer configuration
    context_layers: int = 6  # Layers in context encoder
    denoiser_layers: int = 4  # Layers in diffusion denoiser
    
    # Feed-forward dimension
    d_ff: int = 2048
    
    # Sequence configuration
    max_seq_len: int = 150
    
    # Regularization
    dropout: float = 0.1
    
    # Diffusion configuration
    num_diffusion_steps: int = 100
    variance_schedule: Literal['linear', 'cosine'] = 'cosine'
    beta_start: float = 1e-4
    beta_end: float = 0.02


@dataclass
class ALDTrainingConfig:
    """Training configuration."""
    
    # Data
    train_data_file: str = "./data/helm_sequences_chembl32.txt"
    vocab_file: str = "./data/helm_vocab.json"
    embeddings_dir: str = "./unimol_embeddings"
    data_dir: str = "./data"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Efficient training
    sample_positions_per_seq: int = 5  # Number of positions to sample per sequence
    
    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints/ald"
    
    # Device
    device: str = "cuda"
    
    # Mixed precision
    use_amp: bool = True
    
    # Data loading
    num_workers: int = 4


@dataclass
class ALDGenerationConfig:
    """Generation configuration."""
    
    # Generation parameters
    num_samples: int = 100
    max_length: int = 20
    min_length: Optional[int] = 4
    
    # Diffusion sampling
    use_ddim: bool = True  # Use DDIM for faster sampling
    ddim_steps: int = 50  # Number of DDIM steps (if use_ddim=True)
    
    # Ring bond prediction
    predict_ring_bonds: bool = True
    ring_bond_threshold: float = 0.5
    
    # Token mapping
    use_temperature_sampling: bool = False
    temperature: float = 0.0
    use_freq_weight: bool = True
    freq_weight_scale: float = 0.1
    
    # Output
    output_file: Optional[str] = None
    verbose: bool = True


@dataclass
class ALDConfig:
    """Complete ALD configuration."""
    
    model: ALDModelConfig = field(default_factory=ALDModelConfig)
    training: ALDTrainingConfig = field(default_factory=ALDTrainingConfig)
    generation: ALDGenerationConfig = field(default_factory=ALDGenerationConfig)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict
        
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'generation': asdict(self.generation)
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ALDConfig':
        """Load configuration from JSON file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ALDModelConfig(**config_dict.get('model', {})),
            training=ALDTrainingConfig(**config_dict.get('training', {})),
            generation=ALDGenerationConfig(**config_dict.get('generation', {}))
        )


# Default configurations for different scenarios
def get_default_config() -> ALDConfig:
    """Get default configuration."""
    return ALDConfig()


def get_fast_training_config() -> ALDConfig:
    """Configuration for fast training (debugging)."""
    config = ALDConfig()
    config.model.context_layers = 2
    config.model.denoiser_layers = 2
    config.model.num_diffusion_steps = 50
    config.training.batch_size = 16
    config.training.num_epochs = 10
    return config


def get_large_model_config() -> ALDConfig:
    """Configuration for larger model."""
    config = ALDConfig()
    config.model.d_model = 768
    config.model.n_heads = 12
    config.model.context_layers = 8
    config.model.denoiser_layers = 6
    config.model.d_ff = 3072
    config.model.num_diffusion_steps = 200
    return config


def get_fast_generation_config() -> ALDConfig:
    """Configuration for fast generation."""
    config = ALDConfig()
    config.generation.use_ddim = True
    config.generation.ddim_steps = 25
    return config
