import pandas as pd
from pathlib import Path
import argparse
def extract_helm_sequences(input_csv, output_txt):
    """Extract the HELM column from a CSV file into a text file."""
    print(f"Loading data: {input_csv}")
    
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Loaded {len(df)} rows")
    
    if 'HELM' not in df.columns:
        raise ValueError("CSV file does not contain a HELM column")
    
    helm_sequences = df['HELM'].dropna().astype(str).str.strip()
    
    helm_sequences = helm_sequences[helm_sequences != '']
    
    linear_count = helm_sequences.str.endswith('$$$$').sum()

    cyclic_count = (helm_sequences.str.endswith('$$$') & ~helm_sequences.str.endswith('$$$$')).sum()
    
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for seq in helm_sequences:
            f.write(seq + '\n')
    
    print("\nExtraction complete")
    print(f"Total rows: {len(df):,}")
    print(f"Valid HELM sequences: {len(helm_sequences):,}")
    print(f"  - Linear peptides ($$$$): {linear_count:,}")
    print(f"  - Cyclic peptides (connection info, ending with $$$): {cyclic_count:,}")
    print(f"Output file: {output_path}")
    
    return str(output_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract HELM sequences from a CSV file")
    parser.add_argument("--input_csv", 
                       default="./data/raw/chembl32/biotherapeutics_dict_prot_flt.csv",
                       help="Input CSV path")
    parser.add_argument("--output_txt", 
                       default="./data/processed/helm_sequences_chembl32.txt",
                       help="Output text path")
    
    args = parser.parse_args()
    
    try:
        output_file = extract_helm_sequences(args.input_csv, args.output_txt)
        print(f"\nExtraction succeeded: {output_file}")
        
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    main()
