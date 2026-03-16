"""
Fine-tuning script for Cyclic Peptide Generation with Ring Bond Prediction.

This script fine-tunes a pre-trained ALD model on cyclic peptide data,
adding the capability to predict ring bonds using autoregressive prediction
with contrastive loss for positions and cross-entropy for bond types.

Usage:
    python train_finetune.py
    python train_finetune.py --config configs/finetune.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ald import AutoregressiveLatentDiffusion
from ald.utils.data import HELMDataset, CyclicHELMDataset, HELMCollator
from ald.config import ALDConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune ALD model for cyclic peptides")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Training phase: 1=ring predictor only, 2=end-to-end. None=full training"
    )
    return parser.parse_args()


class CyclicCollator:
    """Collator for cyclic peptide batches with ring bond information."""
    
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate batch with padding and ring bond collection."""
        # Get max sequence length in batch
        max_len = max(len(item['token_ids']) for item in batch)
        
        batch_size = len(batch)
        token_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        ring_bonds = []
        
        for i, item in enumerate(batch):
            seq_len = len(item['token_ids'])
            token_ids[i, :seq_len] = torch.tensor(item['token_ids'])
            mask[i, :seq_len] = True
            ring_bonds.append(item.get('ring_bonds', []))
        
        return {
            'token_ids': token_ids,
            'mask': mask,
            'ring_bonds': ring_bonds
        }


class FinetuneTrainer:
    """Trainer for fine-tuning ALD model on cyclic peptides."""
    
    def __init__(
        self,
        model: AutoregressiveLatentDiffusion,
        train_loader: DataLoader,
        config: ALDConfig,
        vocab: dict,
        phase: Optional[int] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.vocab = vocab
        self.phase = phase
        
        train_cfg = config.training
        
        # Device
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Apply layer freezing based on phase or config
        self._setup_layer_freezing()
        
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"🔧 Trainable parameters: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay
        )
        
        # Learning rate scheduler with warmup + cosine annealing
        total_steps = len(train_loader) * train_cfg.num_epochs
        warmup_steps = train_cfg.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision
        self.scaler = GradScaler() if train_cfg.use_amp else None
        self.use_amp = train_cfg.use_amp
        
        # Checkpoint directory
        self.checkpoint_dir = Path(train_cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config to checkpoint directory
        config.save(self.checkpoint_dir / 'config.json')
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Loss tracking
        self.loss_history = {
            'total_loss': [],
            'diffusion_loss': [],
            'ring_bond_loss': [],
            'ring_position_loss': [],
            'ring_type_loss': [],
            'ce_loss': []
        }
    
    def _setup_layer_freezing(self):
        """Setup layer freezing based on training phase."""
        train_cfg = self.config.training

        if getattr(train_cfg, 'ce_only_finetune', False):
            # CE-only mode: freeze everything except context_encoder + lm_head
            print("📌 CE-only fine-tuning: freezing diffusion engine, ring predictor, embedding")
            for name, param in self.model.named_parameters():
                if 'context_encoder' in name or 'lm_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif self.phase == 1:
            # Phase 1: Only train ring predictor
            print("📌 Phase 1: Freezing all except ring predictor")
            for name, param in self.model.named_parameters():
                if 'ar_ring_predictor' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        elif self.phase == 2:
            # Phase 2: Unfreeze context encoder for end-to-end training
            print("📌 Phase 2: Training ring predictor + context encoder")
            for name, param in self.model.named_parameters():
                if 'ar_ring_predictor' in name or 'context_encoder' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        else:
            # Full training based on config
            if train_cfg.freeze_embedding:
                print("❄️ Freezing embedding layer")
                for param in self.model.embedding_loader.parameters():
                    param.requires_grad = False
            
            if train_cfg.freeze_context_encoder:
                print("❄️ Freezing context encoder")
                for param in self.model.context_encoder.parameters():
                    param.requires_grad = False
            
            if train_cfg.freeze_diffusion:
                print("❄️ Freezing diffusion engine")
                for param in self.model.diffusion_engine.parameters():
                    param.requires_grad = False
    
    def train_step(self, batch) -> Dict[str, torch.Tensor]:
        """Single training step with ring loss."""
        self.model.train()
        
        # Move batch to device
        token_ids = batch['token_ids'].to(self.device)
        mask = batch['mask'].to(self.device)
        ring_bonds = batch['ring_bonds']  # List[List[Dict]], stays on CPU
        
        # Forward pass with ring loss
        if self.use_amp:
            with autocast():
                loss_dict = self.model(
                    token_ids, mask,
                    ring_bonds=ring_bonds,
                    compute_ring_loss=True
                )
                loss = loss_dict['loss']
        else:
            loss_dict = self.model(
                token_ids, mask,
                ring_bonds=ring_bonds,
                compute_ring_loss=True
            )
            loss = loss_dict['loss']
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return loss_dict
    
    def save_checkpoint(self, filename=None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': asdict(self.config),
            'vocab': self.vocab,
            'loss_history': self.loss_history,
            'phase': self.phase
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")
        
        # Also save as 'latest.pt'
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, path, load_optimizer=True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("⚠️ Could not load optimizer state, starting fresh")
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"📂 Loaded checkpoint from {path} (step {self.global_step})")
    
    def train(self):
        """Main training loop."""
        train_cfg = self.config.training
        
        print("\n" + "="*60)
        print("Starting Fine-tuning for Cyclic Peptide Generation")
        if self.phase:
            print(f"Training Phase: {self.phase}")
        print("="*60)
        self.config.print_summary()
        
        start_time = time.time()
        
        for epoch in range(self.epoch, train_cfg.num_epochs):
            self.epoch = epoch
            epoch_losses = {k: 0.0 for k in self.loss_history.keys()}
            num_batches = 0
            
            for batch in self.train_loader:
                loss_dict = self.train_step(batch)
                
                # Accumulate losses
                for key in epoch_losses.keys():
                    if key in loss_dict:
                        val = loss_dict[key]
                        if isinstance(val, torch.Tensor):
                            epoch_losses[key] += val.item()
                        else:
                            epoch_losses[key] += val
                
                num_batches += 1
                
                # Logging
                if self.global_step % train_cfg.log_interval == 0:
                    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    
                    print(f"Step {self.global_step:6d} | "
                          f"Epoch {epoch+1:3d} | "
                          f"Total: {loss_dict['loss'].item():.4f} | "
                          f"Diff: {loss_dict['diffusion_loss'].item():.4f} | "
                          f"Ring: {loss_dict['ring_bond_loss'].item():.4f} "
                          f"(Pos: {loss_dict['ring_position_loss'].item():.3f}, "
                          f"Type: {loss_dict['ring_type_loss'].item():.3f}) | "
                          f"CE: {loss_dict['ce_loss'].item():.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Time: {elapsed:.0f}s")
                
                # Save checkpoint
                if self.global_step % train_cfg.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint()
            
            # End of epoch
            avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            
            # Store in history
            for key, val in avg_epoch_losses.items():
                if key in self.loss_history:
                    self.loss_history[key].append(val)
            
            print(f"\n📊 Epoch {epoch+1} completed:")
            print(f"   Total Loss: {avg_epoch_losses['total_loss']:.4f}")
            print(f"   Diffusion Loss: {avg_epoch_losses['diffusion_loss']:.4f}")
            print(f"   Ring Bond Loss: {avg_epoch_losses['ring_bond_loss']:.4f}")
            print(f"   Ring Position Loss: {avg_epoch_losses['ring_position_loss']:.4f}")
            print(f"   Ring Type Loss: {avg_epoch_losses['ring_type_loss']:.4f}")
            print(f"   CE Loss: {avg_epoch_losses['ce_loss']:.4f}")
            print()
            
            # Save loss to log file
            log_path = self.checkpoint_dir / "training_log.txt"
            with open(log_path, "a") as f:
                f.write(f"Epoch {epoch+1}: " + 
                        ", ".join([f"{k}={v:.6f}" for k, v in avg_epoch_losses.items()]) +
                        f", Time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Save end-of-epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        print("\n" + "="*60)
        print("Fine-tuning Complete!")
        print("="*60)
        
        # Save final model
        self.save_checkpoint("final_model.pt")


def main():
    """Main function."""
    args = parse_args()
    
    config_path = Path(args.config)
    
    # Load config from file
    print(f"📄 Loading config from: {config_path}")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("   Creating default finetune config...")
        # Create default finetune config
        config = ALDConfig()
        config.training.finetune_mode = True
        config.training.cyclic_only = True
        config.training.learning_rate = 5e-5
        config.training.num_epochs = 50
        config.save(str(config_path))
    
    config = ALDConfig.load(str(config_path))
    
    # Override phase if specified
    phase = args.phase
    
    # Load vocab
    with open(config.training.vocab_file, 'r') as f:
        vocab = json.load(f)
    print(f"📚 Loaded vocabulary with {len(vocab)} tokens")
    
    # Create dataset (cyclic peptides only)
    if config.training.cyclic_only:
        dataset = CyclicHELMDataset(
            data_file=config.training.train_data_file,
            vocab_file=config.training.vocab_file,
            max_seq_len=config.model.max_seq_len
        )
        print(f"📊 Loaded {len(dataset)} cyclic peptide sequences")
    else:
        dataset = HELMDataset(
            data_file=config.training.train_data_file,
            vocab_file=config.training.vocab_file,
            max_seq_len=config.model.max_seq_len,
            cyclic_only=True
        )
        print(f"📊 Loaded {len(dataset)} sequences")
    
    # Create dataloader with cyclic collator
    collator = CyclicCollator(vocab['<PAD>'])
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    # Create model
    model = AutoregressiveLatentDiffusion(vocab=vocab, config=config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Created model with {total_params/1e6:.1f}M parameters")
    
    # Load pretrained checkpoint if specified
    if config.training.finetune_mode and config.training.pretrained_checkpoint:
        pretrained_path = Path(config.training.pretrained_checkpoint)
        if pretrained_path.exists():
            print(f"📂 Loading pretrained model from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # Load state dict, allowing missing keys (for ring predictor)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ Pretrained model loaded successfully")
        else:
            print(f"⚠️ Pretrained checkpoint not found: {pretrained_path}")
            print("   Starting from scratch")
    
    # Create trainer
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        vocab=vocab,
        phase=phase
    )
    
    # Check for resume
    latest_checkpoint = Path(config.training.checkpoint_dir) / 'latest.pt'
    if latest_checkpoint.exists():
        print(f"\n🔄 Found existing checkpoint, resuming training...")
        trainer.load_checkpoint(latest_checkpoint)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
