import re
from collections import Counter

def classify_topology(helm_seq):
    """
    分类肽拓扑：
    - linear: 无连接
    - cyclic: 仅首尾成环 (1 <-> length)
    - q_type: 第一个或最后一个单体与中间单体成环
    - other: 其他类型环肽
    """
    match = re.search(r'PEPTIDE1\{([^}]+)\}', helm_seq)
    if not match:
        return 'linear', []
    
    monomers = match.group(1).split('.')
    length = len(monomers)
    
    # 提取所有连接 (pos1:R*-pos2:R*)
    connections = re.findall(r'(\d+):R\d+-(\d+):R\d+', helm_seq)
    if not connections:
        return 'linear', monomers
    
    has_head_tail = False
    has_q_type = False
    has_other = False
    
    for p1_str, p2_str in connections:
        p1, p2 = int(p1_str), int(p2_str)
        if p1 > p2:
            p1, p2 = p2, p1
        
        if p1 == 1 and p2 == length:
            has_head_tail = True
        elif (p1 == 1 and 1 < p2 < length) or (1 < p1 < length and p2 == length):
            has_q_type = True
        else:
            has_other = True
    
    # 优先级: Q型 > Other > Cyclic
    if has_q_type:
        return 'q_type', monomers
    elif has_other:
        return 'other', monomers
    elif has_head_tail:
        return 'cyclic', monomers
    return 'linear', monomers

def analyze_lengths(file_path):
    lengths = []
    x_monomer_seq_count = 0
    
    # 拓扑计数器
    topology_counts = {'linear': 0, 'cyclic': 0, 'q_type': 0, 'other': 0}

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            topo_type, monomers = classify_topology(line)
            if monomers:
                topology_counts[topo_type] += 1
                lengths.append(len(monomers))
                
                # Check for X-starting monomers
                if any(m.strip('[]').startswith('X') for m in monomers):
                    x_monomer_seq_count += 1
    
    total = len(lengths)
    if total == 0:
        print("No sequences found.")
        return
    
    print(f"\nFile: {file_path}")
    print(f"Total sequences: {total}")
    print("=" * 50)
    print("TOPOLOGY CLASSIFICATION")
    print("=" * 50)
    print(f"Linear:                 {topology_counts['linear']:<6} ({100*topology_counts['linear']/total:.2f}%)")
    print(f"Cyclic (Head-to-Tail):  {topology_counts['cyclic']:<6} ({100*topology_counts['cyclic']/total:.2f}%)")
    print(f"Q-Type (Terminal-Mid):  {topology_counts['q_type']:<6} ({100*topology_counts['q_type']/total:.2f}%)")
    print(f"Other Cyclic:           {topology_counts['other']:<6} ({100*topology_counts['other']/total:.2f}%)")
    
    print("\n" + "=" * 50)
    print("LENGTH STATISTICS")
    print("=" * 50)
    print(f"Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/total:.2f}")
    print(f"\nSequences with 'X' monomers: {x_monomer_seq_count} ({100*x_monomer_seq_count/total:.2f}%)")
    
    print(f"\n{'Length >':<10} | {'Count':<10} | {'%':<10}")
    print("-" * 35)
    length_counts = Counter(lengths)
    for t in [10, 15, 20, 25, 30, 35, 40, 45]:
        count = sum(c for l, c in length_counts.items() if l > t)
        print(f"> {t:<8} | {count:<10} | {100*count/total:.2f}%")

    print("\nTop 10 most common lengths:")
    for length, count in length_counts.most_common(10):
        print(f"  {length}: {count} ({100*count/total:.2f}%)")

if __name__ == "__main__":
    file_path = "./data/helm_sequences_cycpeptmpdb.txt"
    analyze_lengths(file_path)