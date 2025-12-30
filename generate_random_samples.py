import json
import random
import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).parent
    vocab_path = root / "data/helm_vocab.json"
    monomer_path = root / "data/monomer_library.csv"
    output_dir = root / "chembl32_samples"
    output_path = output_dir / "random_generated_samples.txt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    df = pd.read_csv(monomer_path)
    
    start_pool = []
    middle_pool = []
    end_pool = []
    
    for _, row in df.iterrows():
        symbol = row['Symbol']
        if symbol not in vocab:
            continue
            
        r1 = str(row.get('R1', '-')).strip()
        r2 = str(row.get('R2', '-')).strip()
        
        has_r1 = r1 not in ('-', 'nan', '')
        has_r2 = r2 not in ('-', 'nan', '')
        
        if has_r2:
            start_pool.append(symbol)
        if has_r1 and has_r2:
            middle_pool.append(symbol)
        if has_r1:
            end_pool.append(symbol)
            
    samples = []
    for _ in range(5000):
        length = random.randint(4, 20)
        
        seq = []
        seq.append(random.choice(start_pool))
        
        for _ in range(length - 2):
            seq.append(random.choice(middle_pool))
            
        seq.append(random.choice(end_pool))
        
        helm = f"PEPTIDE1{'{'}{'.'.join(seq)}{'}'}$$$$"
        samples.append(helm)
        
    with open(output_path, 'w') as f:
        f.write('\n'.join(samples))

if __name__ == "__main__":
    main()
