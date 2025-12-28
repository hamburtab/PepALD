import pandas as pd
import re
from pathlib import Path
import argparse


def normalize_helm_sequence(helm_seq):
    """
    Normalize HELM sequence by converting PEPTIDE numbers to 1
    Example: PEPTIDE15{...} -> PEPTIDE1{...}
    """
    if not isinstance(helm_seq, str):
        return helm_seq
    
    # Replace PEPTIDE + number with PEPTIDE1
    normalized = re.sub(r'PEPTIDE\d+', 'PEPTIDE1', helm_seq)
    return normalized


def extract_helm_sequences(input_csv, output_txt):
    """from CSV file directly extract all data in HELM column to txt file"""
    
    print(f'Loading data: {input_csv}')
    
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Successfully read {len(df)} rows of data")

    if 'HELM' not in df.columns:
        raise ValueError("HELM column not found in CSV file")

    helm_sequences = df['HELM'].dropna().astype(str).str.strip()
    
    helm_sequences = helm_sequences[helm_sequences != '']
    
    print(f"Original valid HELM sequences: {len(helm_sequences):,}")
    
    # Normalize HELM sequences (convert PEPTIDE numbers to 1)
    helm_sequences = helm_sequences.apply(normalize_helm_sequence)
    
    # Remove duplicate sequences
    unique_helm_sequences = helm_sequences.drop_duplicates()
    
    print(f"Unique HELM sequences after deduplication: {len(unique_helm_sequences):,}")
    print(f"Removed {len(helm_sequences) - len(unique_helm_sequences):,} duplicate sequences")
    
    linear_count = unique_helm_sequences.str.endswith('$$$$').sum()

    cyclic_count = (unique_helm_sequences.str.endswith('$$$') & ~unique_helm_sequences.str.endswith('$$$$')).sum()
    
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq in unique_helm_sequences:
            f.write(seq + '\n')

    print(f"CycPeptMPDB HELM extraction complete")
    print(f"Total rows in input file: {len(df):,}")
    print(f"Valid HELM sequences: {len(unique_helm_sequences):,}")
    print(f"  - Linear peptides ($$$$): {linear_count:,}")
    print(f"  - Cyclic peptides (with connection info, ending with $$$): {cyclic_count:,}")
    print(f"Output file: {output_path}")
    
    return str(output_path)


def main():
    """main function"""

    parser = argparse.ArgumentParser(description="Extract HELM sequences from CSV to txt file")

    parser.add_argument("--input_csv", 
                       default="./data/CycPeptMPDB/CycPeptMPDB_Peptide_All.csv",
                       help="Path to the input CSV file")
    parser.add_argument("--output_txt", 
                       default="./data/helm_sequences_cycpeptmpdb.txt",
                       help="Path to the output txt file")

    args = parser.parse_args()
    
    try:
        output_file = extract_helm_sequences(args.input_csv, args.output_txt)
        print(f"\nExtraction successful! Output file: {output_file}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
