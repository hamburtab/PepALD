"""
Configuration for Autoregressive Latent Diffusion (ALD) model.

This is the SINGLE SOURCE OF TRUTH for all model and training parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from pathlib import Path
import json


@dataclass
class ALDModelConfig:
    """Model architecture configuration."""

    # Model ablation variant. The default preserves the full PepALD model.
    model_variant: Literal['ald', 'lm_only'] = 'ald'
    
    # Embedding dimensions
    embedding_dim: int = 512  # Will be auto-updated from Uni-Mol embeddings
    d_model: int = 512

    # Chemical embedding ablation
    chememb_mode: Literal['original', 'shuffled'] = 'original'
    chememb_shuffle_seed: int = 42
    
    # Attention configuration
    n_heads: int = 8
    
    # Layer configuration
    context_layers: int = 6
    denoiser_layers: int = 4
    
    # Feed-forward dimension
    d_ff: int = 2048
    
    # Sequence configuration
    max_seq_len: int = 45
    
    # Regularization
    dropout: float = 0.1
    
    # Diffusion configuration
    num_diffusion_steps: int = 100
    variance_schedule: Literal['linear', 'cosine'] = 'cosine'
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    # Ring predictor configuration
    ring_feature_mode: Literal['context_rsite', 'context_only'] = 'context_rsite'
    r_group_dim: int = 512  # R-group embedding dimension
    ring_hidden_dim: int = 256  # Hidden dimension for ring predictor
    num_ring_types: int = 4  # Number of ring bond types (R3R3, R1R2, R1R3, R3R2)


@dataclass
class ALDTrainingConfig:
    """Training configuration."""
    
    # Data paths
    train_data_file: str = "./data/processed/helm_sequences_chembl32.txt"
    vocab_file: str = "./data/processed/helm_vocab.json"
    embeddings_dir: str = "./data/processed/unimol_embeddings"
    data_dir: str = "./data/processed"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Efficient training: sample positions per sequence
    # Set to max_seq_len for full training
    sample_positions: int = 45
    
    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 1000
    save_epoch_interval: int = 1
    checkpoint_dir: str = "./checkpoints/ald"
    
    # Device
    device: str = "cuda"
    
    # Mixed precision
    use_amp: bool = False
    
    # Data loading
    num_workers: int = 4
    
    # Fine-tuning configuration
    finetune_mode: bool = False
    pretrained_checkpoint: Optional[str] = None
    cyclic_only: bool = False
    
    # Fine-tuning phases (loss weights)
    ring_loss_weight: float = 0.5
    ce_loss_weight: float = 0.5
    
    # CE-only fine-tuning: only train ce_loss, freeze diffusion & ring predictor
    ce_only_finetune: bool = False

    # Layer freezing for fine-tuning
    freeze_embedding: bool = False
    freeze_context_encoder: bool = False
    freeze_diffusion: bool = False


@dataclass
class ALDGenerationConfig:
    """Generation configuration."""
    
    # Checkpoint path for inference
    checkpoint_path: str = "./checkpoints/ald/checkpoint_epoch_99.pt"
    
    # Generation parameters
    num_samples: int = 100
    max_length: int = 20
    min_length: Optional[int] = 4
    
    # Diffusion sampling
    use_ddim: bool = True
    ddim_steps: int = 50
    
    # Ring bond prediction
    predict_ring_bonds: bool = True
    cyclization_mode: Literal['predict_ring', 'force_head_tail'] = 'predict_ring'
    ring_bond_threshold: float = 0.5
    ring_top_k: int = 1  # Number of top candidates to consider for ring prediction
    
    # Token mapping
    use_embedding_norm: bool = True  # Use cosine distance
    # Enforce backbone-connectivity masks during token selection:
    # first residue has R2, middle residues have R1+R2, last residue has R1.
    # Disable this for an unconstrained-generation ablation.
    enforce_r1r2_constraints: bool = True
    lambda_gpt: float = 0.0
    mapping_sample: bool = False
    mapping_top_k: int = 8
    mapping_top_p: float = 1.0
    mapping_temperature: float = 1.0
    mapping_frequency_penalty: float = 0.0
    history_embedding_mode: Literal['token', 'latent'] = 'token'
    
    # Output
    output_file: Optional[str] = None
    verbose: bool = True
    
    # Random seed (None for random)
    seed: Optional[int] = None


@dataclass
class ALDConfig:
    """
    Complete ALD configuration.
    
    This is the SINGLE SOURCE OF TRUTH for all parameters.
    
    Usage:
        # Load default config
        config = ALDConfig()
        
        # Load from JSON file
        config = ALDConfig.load('config.json')
        
        # Modify and save
        config.model.d_model = 768
        config.save('new_config.json')
    """
    
    model: ALDModelConfig = field(default_factory=ALDModelConfig)
    training: ALDTrainingConfig = field(default_factory=ALDTrainingConfig)
    generation: ALDGenerationConfig = field(default_factory=ALDGenerationConfig)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'generation': asdict(self.generation)
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ALDConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            model=ALDModelConfig(**config_dict.get('model', {})),
            training=ALDTrainingConfig(**config_dict.get('training', {})),
            generation=ALDGenerationConfig(**config_dict.get('generation', {}))
        )
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("\n" + "="*60)
        print("ALD Configuration Summary")
        print("="*60)
        
        print("\n[Model]")
        print(f"  model_variant:      {self.model.model_variant}")
        print(f"  d_model:            {self.model.d_model}")
        print(f"  chememb_mode:       {self.model.chememb_mode}")
        print(f"  chememb_seed:       {self.model.chememb_shuffle_seed}")
        print(f"  ring_features:      {self.model.ring_feature_mode}")
        print(f"  n_heads:            {self.model.n_heads}")
        print(f"  context_layers:     {self.model.context_layers}")
        print(f"  denoiser_layers:    {self.model.denoiser_layers}")
        print(f"  d_ff:               {self.model.d_ff}")
        print(f"  max_seq_len:        {self.model.max_seq_len}")
        print(f"  dropout:            {self.model.dropout}")
        print(f"  diffusion_steps:    {self.model.num_diffusion_steps}")
        print(f"  variance_schedule:  {self.model.variance_schedule}")
        
        print("\n[Training]")
        print(f"  batch_size:         {self.training.batch_size}")
        print(f"  learning_rate:      {self.training.learning_rate}")
        print(f"  num_epochs:         {self.training.num_epochs}")
        print(f"  warmup_steps:       {self.training.warmup_steps}")
        print(f"  sample_positions:   {self.training.sample_positions}")
        print(f"  device:             {self.training.device}")
        print(f"  use_amp:            {self.training.use_amp}")
        
        print("\n[Generation]")
        print(f"  use_ddim:           {self.generation.use_ddim}")
        print(f"  ddim_steps:         {self.generation.ddim_steps}")
        print(f"  use_embedding_norm: {self.generation.use_embedding_norm}")
        print(f"  enforce_r1r2:       {self.generation.enforce_r1r2_constraints}")
        print(f"  lambda_gpt:         {self.generation.lambda_gpt}")
        print(f"  mapping_sample:     {self.generation.mapping_sample}")
        print(f"  history_mode:       {self.generation.history_embedding_mode}")
        print("="*60 + "\n")


# ============================================================
# Preset Configurations
# ============================================================

def get_default_config() -> ALDConfig:
    """Get default configuration for ChEMBL32 training."""
    return ALDConfig()


def get_debug_config() -> ALDConfig:
    """Configuration for quick debugging."""
    config = ALDConfig()
    config.model.context_layers = 2
    config.model.denoiser_layers = 2
    config.model.num_diffusion_steps = 20
    config.training.batch_size = 8
    config.training.num_epochs = 2
    config.training.log_interval = 10
    config.training.save_interval = 50
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
    """Configuration for fast generation with DDIM."""
    config = ALDConfig()
    config.generation.use_ddim = True
    config.generation.ddim_steps = 25
    return config
