import os
import numpy as np
import re

def get_helm_stats(helm_seq):
    try:
        # Extract the sequence part between first { and first }
        start = helm_seq.find('{') + 1
        end = helm_seq.find('}')
        if start == 0 or end == -1:
            return 0, False
        
        seq_part = helm_seq[start:end]
        if not seq_part:
            return 0, False
            
        # Split by '.' to get monomers
        monomers = seq_part.split('.')
        length = len(monomers)
        
        # Check for connections
        # Look for pattern like "1:R1-11:R2" in the part after the sequence
        rest = helm_seq[end+1:]
        
        # Regex to find all connections: number:Rnumber-number:Rnumber
        # This handles standard HELM connection format
        connections = re.findall(r'(\d+):R\d+-(\d+):R\d+', rest)
        
        has_any_ring = len(connections) > 0
        is_head_to_tail = False
        
        for c in connections:
            idx1, idx2 = int(c[0]), int(c[1])
            # Check if it connects 1 and length (head-to-tail)
            if (idx1 == 1 and idx2 == length) or (idx1 == length and idx2 == 1):
                is_head_to_tail = True
                break
                
        return length, has_any_ring, is_head_to_tail
    except Exception as e:
        # print(f"Error parsing HELM: {helm_seq[:50]}... - {e}")
        return 0, False, False

def analyze_file(file_path, dataset_name):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\nAnalyzing {dataset_name}...")
    lengths = []
    total_cyclic_count = 0      # Count of sequences with ANY ring
    head_to_tail_count = 0      # Count of sequences with head-to-tail ring
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                length, has_ring, is_head_to_tail = get_helm_stats(line)
                if length > 0:
                    lengths.append(length)
                    if has_ring:
                        total_cyclic_count += 1
                    if is_head_to_tail:
                        head_to_tail_count += 1
    
    if lengths:
        avg_len = np.mean(lengths)
        min_len = np.min(lengths)
        max_len = np.max(lengths)
        median_len = np.median(lengths)
        
        total_seqs = len(lengths)
        cyclic_ratio = total_cyclic_count / total_seqs
        
        # Ratio of head-to-tail among ALL sequences
        ht_ratio_all = head_to_tail_count / total_seqs
        
        # Ratio of head-to-tail among CYCLIC sequences
        ht_ratio_cyclic = head_to_tail_count / total_cyclic_count if total_cyclic_count > 0 else 0
        
        print(f"  Total sequences: {total_seqs}")
        print(f"  Average length: {avg_len:.2f}")
        print(f"  Median length: {median_len:.2f}")
        print(f"  Min length: {min_len}")
        print(f"  Max length: {max_len}")
        print(f"  ------------------------------------------------")
        print(f"  Sequences with ANY ring: {total_cyclic_count} ({cyclic_ratio:.2%})")
        print(f"  Head-to-tail cyclic: {head_to_tail_count}")
        print(f"    - Ratio in ALL sequences: {ht_ratio_all:.2%}")
        print(f"    - Ratio in CYCLIC sequences: {ht_ratio_cyclic:.2%}")
    else:
        print("  No valid sequences found.")

def main():
    # Define paths relative to this script or absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    chembl_path = os.path.join(data_dir, 'helm_sequences_chembl32.txt')
    cycpept_path = os.path.join(data_dir, 'helm_sequences_cycpeptmpdb.txt')
    
    analyze_file(chembl_path, "ChEMBL32")
    analyze_file(cycpept_path, "CycPeptMPDB")

if __name__ == "__main__":
    main()
