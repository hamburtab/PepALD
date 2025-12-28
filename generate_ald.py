"""
Generation script for Autoregressive Latent Diffusion (ALD) model.

Usage:
    python generate_ald.py --checkpoint ./checkpoints/ald/final_model.pt --num_samples 100
    python generate_ald.py --checkpoint ./checkpoints/ald/final_model.pt --output samples.txt --ddim
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ald import AutoregressiveLatentDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description='Generate peptides using ALD model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='./data/helm_vocab.json',
                        help='Path to vocabulary file')
    parser.add_argument('--embeddings', type=str, default='./unimol_embeddings',
                        help='Path to Uni-Mol embeddings directory')
    
    # Generation arguments
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of sequences to generate')
    parser.add_argument('--max_length', type=int, default=45,
                        help='Maximum sequence length')
    parser.add_argument('--min_length', type=int, default=5,
                        help='Minimum sequence length')
    
    # Sampling method
    parser.add_argument('--ddim', action='store_true',
                        help='Use DDIM sampling (faster)')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM steps')
    
    # Ring bond prediction
    parser.add_argument('--no_ring_bonds', action='store_true',
                        help='Disable ring bond prediction')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation (currently only 1 supported)')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress')
    
    return parser.parse_args()


def load_model(args):
    """Load model from checkpoint."""
    
    # Load vocabulary
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Load checkpoint to get model config
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Create model with config from checkpoint
    model = AutoregressiveLatentDiffusion(
        vocab=vocab,
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        context_layers=config.get('context_layers', 6),
        denoiser_layers=config.get('denoiser_layers', 4),
        d_ff=config.get('d_model', 512) * 4,
        max_seq_len=config.get('max_seq_len', 150),
        dropout=0.0,  # No dropout during inference
        num_diffusion_steps=config.get('diffusion_steps', 100),
        variance_schedule='cosine',
        embeddings_dir=args.embeddings,
        data_dir='./data'
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")
    
    return model


def generate(model, args):
    """Generate peptide sequences."""
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"\nGenerating {args.num_samples} sequences...")
    print(f"  Length range: {args.min_length} - {args.max_length}")
    print(f"  Sampling: {'DDIM' if args.ddim else 'DDPM'}")
    if args.ddim:
        print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Ring bonds: {'disabled' if args.no_ring_bonds else 'enabled'}")
    
    start_time = time.time()
    
    # Generate sequences
    helm_sequences = []
    
    for i in range(args.num_samples):
        if args.verbose:
            print(f"\nGenerating sequence {i+1}/{args.num_samples}...")
        
        results = model.sample(
            num_samples=1,
            max_length=args.max_length,
            min_length=args.min_length,
            device=device,
            use_ddim=args.ddim,
            ddim_steps=args.ddim_steps,
            predict_ring_bonds=not args.no_ring_bonds,
            verbose=args.verbose
        )
        
        # Decode to HELM
        result = results[0]
        helm = model.decode_to_helm(
            result['tokens'],
            result.get('ring_connections', [])
        )
        helm_sequences.append(helm)
        
        if not args.verbose and (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{args.num_samples}")
    
    elapsed = time.time() - start_time
    print(f"\nGeneration completed in {elapsed:.1f}s ({elapsed/args.num_samples:.2f}s per sequence)")
    
    return helm_sequences


def analyze_sequences(sequences):
    """Analyze generated sequences."""
    
    from ald.utils.topology import HELMTopologyAnalyzer
    
    analyzer = HELMTopologyAnalyzer()
    
    lengths = []
    types = {'linear': 0, 'cyclic': 0, 'q_type': 0}
    
    for helm in sequences:
        parsed = analyzer.parse_helm_sequence(helm)
        lengths.append(len(parsed['monomers']))
        types[parsed['peptide_type']] = types.get(parsed['peptide_type'], 0) + 1
    
    print("\n" + "="*60)
    print("Generated Sequences Analysis")
    print("="*60)
    print(f"Total sequences: {len(sequences)}")
    print(f"Length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    print(f"Types: {types}")
    
    # Show some examples
    print("\nExample sequences:")
    for i, helm in enumerate(sequences[:5]):
        print(f"  {i+1}. {helm}")


def main():
    args = parse_args()
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(args)
    
    # Generate
    sequences = generate(model, args)
    
    # Analyze
    analyze_sequences(sequences)
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for helm in sequences:
                f.write(helm + '\n')
        
        print(f"\nSaved {len(sequences)} sequences to {output_path}")
    else:
        print("\nGenerated sequences (use --output to save):")
        for i, helm in enumerate(sequences):
            print(f"{i+1}. {helm}")


if __name__ == '__main__':
    main()
