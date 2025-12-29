"""HELM Topology Analyzer - parses HELM sequences and extracts structural info."""

import re
from typing import Dict, List, Optional, Tuple


class HELMTopologyAnalyzer:
    """Analyzes HELM sequence topology and structure."""
    
    BOND_TYPES = ['R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def parse_helm_sequence(self, helm_sequence: str) -> Dict:
        """Parse HELM sequence and extract structural information."""
        parts = helm_sequence.split('$')
        sequence_part = parts[0]
        
        connection_part = ""
        for part in parts[1:]:
            if part and 'PEPTIDE' in part and ':R' in part:
                connection_part = part
                break
        
        peptide_match = re.search(r'PEPTIDE\d+\{([^}]+)\}', sequence_part)
        if not peptide_match:
            return {'peptide_type': 'linear', 'sequence': '', 'monomers': [], 
                    'connections': [], 'raw_helm': helm_sequence}
        
        sequence = peptide_match.group(1)
        monomers = self._parse_monomers(sequence)
        connections = self._parse_connections(connection_part)
        peptide_type = self._determine_peptide_type(len(monomers), connections)
        
        return {
            'peptide_type': peptide_type,
            'sequence': sequence,
            'monomers': monomers,
            'connections': connections,
            'raw_helm': helm_sequence
        }
    
    def _parse_monomers(self, sequence: str) -> List[str]:
        """Parse monomer sequence into list of symbols."""
        if not sequence:
            return []
        monomers = []
        for m in sequence.split('.'):
            if m.startswith('[') and m.endswith(']'):
                m = m[1:-1]
            monomers.append(m)
        return monomers
    
    def _parse_connections(self, connection_part: str) -> List[Dict]:
        """Parse connection information from HELM string."""
        if not connection_part:
            return []
        connections = []
        for match in re.findall(r'(\d+):R(\d+)-(\d+):R(\d+)', connection_part):
            pos1, r1, pos2, r2 = match
            connections.append({'pos1': int(pos1), 'r1': int(r1), 'pos2': int(pos2), 'r2': int(r2)})
        return connections
    
    def _determine_peptide_type(self, seq_length: int, connections: List[Dict]) -> str:
        """Determine peptide topology type (linear/cyclic/q_type)."""
        if not connections:
            return 'linear'
        
        cyclic, q_type = 0, 0
        for conn in connections:
            pos1, pos2 = conn['pos1'], conn['pos2']
            if (pos1 == 1 and pos2 == seq_length) or (pos1 == seq_length and pos2 == 1):
                cyclic += 1
            else:
                q_type += 1
        
        if cyclic > 0 and q_type == 0:
            return 'cyclic'
        return 'q_type' if q_type > 0 else 'linear'
    
    def get_ring_bond_matrix(self, helm_sequence: str) -> Optional[Tuple[List[List[int]], List[str]]]:
        """Get ring bond info as matrix. Returns (bond_matrix, monomers) or None."""
        parsed = self.parse_helm_sequence(helm_sequence)
        monomers, connections = parsed['monomers'], parsed['connections']
        
        if not monomers:
            return None
        
        n = len(monomers)
        bond_matrix = [[0] * n for _ in range(n)]
        
        for conn in connections:
            pos1, pos2 = conn['pos1'] - 1, conn['pos2'] - 1
            r1, r2 = conn['r1'], conn['r2']
            
            if 0 <= pos1 < n and 0 <= pos2 < n:
                r_link = f"R{r1}R{r2}"
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    r_link = f"R{r2}R{r1}"
                if r_link in self.BOND_TYPES:
                    bond_matrix[pos1][pos2] = self.BOND_TYPES.index(r_link) + 1
        
        return bond_matrix, monomers
    
    def extract_ring_info(self, helm_sequence: str) -> Optional[Dict]:
        """Extract ring bond info for training. Returns flattened upper triangular."""
        result = self.get_ring_bond_matrix(helm_sequence)
        if result is None:
            return None
        
        bond_matrix, monomers = result
        n = len(monomers)
        bond_array = [bond_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        
        return {'sequence_length': n, 'bond_array': bond_array, 'num_pairs': len(bond_array)}
