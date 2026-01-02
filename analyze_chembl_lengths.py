import re
from collections import Counter

def get_monomer_count(helm_string):
    # Extract the content inside PEPTIDE1{...}
    match = re.search(r'PEPTIDE1\{(.*?)\}', helm_string)
    if match:
        sequence = match.group(1)
        # Split by dot to get monomers
        monomers = sequence.split('.')
        return len(monomers)
    return 0

def analyze_lengths(file_path):
    lengths = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            length = get_monomer_count(line)
            if length > 0:
                lengths.append(length)
    
    total_sequences = len(lengths)
    length_counts = Counter(lengths)
    
    target_lengths = [10, 15, 20, 25, 30, 35, 40, 45]
    
    print(f"Total sequences: {total_sequences}")
    print("-" * 30)
    print(f"{'Length':<10} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 30)
    
    for threshold in target_lengths:
        count = sum(c for l, c in length_counts.items() if l > threshold)
        percentage = (count / total_sequences) * 100
        print(f"> {threshold:<8} | {count:<10} | {percentage:.2f}%")
        
    print("-" * 30)
    # Also print some general stats
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {sum(lengths) / total_sequences:.2f}")
    
    # Print top 10 most common lengths
    print("\nTop 10 most common lengths:")
    for length, count in length_counts.most_common(10):
        percentage = (count / total_sequences) * 100
        print(f"Length {length}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    file_path = "/root/New-HELM-Diffusion/data/helm_sequences_chembl32.txt"
    analyze_lengths(file_path)
