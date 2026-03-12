import csv, json, re
from collections import Counter

monomer_counter = Counter()
total_seqs = 0
cyclic_count = 0

with open('data/prior_data_sorted_1000.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        helm = row.get('HELM', '').strip()
        if not helm:
            continue
        total_seqs += 1
        if 'PEPTIDE1,PEPTIDE1' in helm:
            cyclic_count += 1
        match = re.search(r'PEPTIDE1\{(.+?)\}', helm)
        if match:
            chain = match.group(1)
            monomers = chain.split('.')
            for m in monomers:
                m_clean = m.strip('[]')
                monomer_counter[m_clean] += 1

print(f'Total sequences: {total_seqs}')
print(f'Cyclic sequences: {cyclic_count}')
print(f'Unique monomers: {len(monomer_counter)}')
print(f'Top 20 monomers:')
for m, c in monomer_counter.most_common(20):
    print(f'  {m}: {c}')

# Check if X2451 and X1482 appear
print(f'\nX2451 count: {monomer_counter.get("X2451", 0)}')
print(f'X1482 count: {monomer_counter.get("X1482", 0)}')

# Check vocab
with open('data/helm_vocab.json') as f:
    vocab = json.load(f)
print(f'\nX2451 in vocab: {vocab.get("X2451", "NOT FOUND")}')
print(f'X1482 in vocab: {vocab.get("X1482", "NOT FOUND")}')

# Check how many monomers from training data are NOT in vocab
missing = 0
for m in monomer_counter:
    if m not in vocab:
        missing += 1
        if missing <= 5:
            print(f'Missing from vocab: {m}')
print(f'Total monomers missing from vocab: {missing}')
