"""
Validity evaluation for generated HELM sequences.
This script evaluates the validity and uniqueness of HELM sequences
using the helm2smiles utility functions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pepar_diff.utils.helm import is_helm_valid, get_validity, get_uniqueness


def load_helm_sequences(file_path: str) -> list:
    """
    Load HELM sequences from a text file.
    
    Args:
        file_path: Path to the file containing HELM sequences (one per line)
    
    Returns:
        List of HELM sequences
    """
    with open(file_path, 'r') as f:
        helms = [line.strip() for line in f if line.strip()]
    return helms


def evaluate_validity(helm_file: str, verbose: bool = True):
    """
    Evaluate the validity and uniqueness of HELM sequences.
    
    Args:
        helm_file: Path to the file containing HELM sequences
        verbose: If True, print detailed results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load sequences
    helms = load_helm_sequences(helm_file)
    
    if len(helms) == 0:
        print("No HELM sequences found in the file.")
        return None
    
    # Calculate validity
    validity = get_validity(helms)
    
    # Calculate uniqueness (only for valid sequences)
    valid_helms = [x for x in helms if is_helm_valid(x)]
    uniqueness = get_uniqueness(helms) if len(valid_helms) > 0 else 0.0
    
    # Prepare results
    results = {
        'total_sequences': len(helms),
        'valid_sequences': len(valid_helms),
        'unique_valid_sequences': len(set(valid_helms)),
        'validity': validity,
        'uniqueness': uniqueness
    }
    
    # Print results
    if verbose:
        print("=" * 60)
        print("HELM Sequence Evaluation Results")
        print("=" * 60)
        print(f"Total sequences:          {results['total_sequences']}")
        print(f"Valid sequences:          {results['valid_sequences']}")
        print(f"Unique valid sequences:   {results['unique_valid_sequences']}")
        print(f"Validity:                 {results['validity']:.2%}")
        print(f"Uniqueness:               {results['uniqueness']:.2%}")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Default sample file path
    sample_file = str(PROJECT_ROOT / "outputs" / "samples" / "case1" / "generated" / "helm_dpo_samples.txt")
    
    # Run evaluation
    evaluate_validity(sample_file)
