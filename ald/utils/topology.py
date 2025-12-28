"""
HELM Topology Analyzer.

Parses HELM sequences and extracts structural information including:
    - Peptide type (linear, cyclic, q_type)
    - Sequence of monomers
    - Ring bond connections

This is adapted from the original helm_topology_analyzer.py.
"""

import re
from typing import Dict, List, Optional, Tuple


class HELMTopologyAnalyzer:
    """
    Analyzes HELM sequence topology and structure.
    
    HELM (Hierarchical Editing Language for Macromolecules) notation:
        - Sequence part: PEPTIDE1{monomer1.monomer2....}
        - Connection part: PEPTIDE1,PEPTIDE1,pos1:Rx-pos2:Ry
        - Annotations: Additional metadata
    
    Example:
        Linear: PEPTIDE1{A.C.G.F.K}$$$$
        Cyclic: PEPTIDE1{C.G.C.R.K}$PEPTIDE1,PEPTIDE1,6:R3-1:R3$$$
        Q-type: PEPTIDE1{A.C.G.C.K}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$
    """
    
    # Ring bond type mapping
    BOND_TYPES = ['R3R3', 'R1R2', 'R1R3', 'R3R2']
    
    def __init__(self):
        pass
    
    def parse_helm_sequence(self, helm_sequence: str) -> Dict:
        """
        Parse a HELM sequence and extract structural information.
        
        Args:
            helm_sequence: Full HELM string
            
        Returns:
            Dictionary with:
                - peptide_type: 'linear', 'cyclic', or 'q_type'
                - sequence: The monomer sequence string
                - monomers: List of monomer symbols
                - connections: List of connection dictionaries
                - raw_helm: Original HELM string
        """
        # Split by $ delimiter
        parts = helm_sequence.split('$')
        sequence_part = parts[0]
        
        # Find connection part
        connection_part = ""
        for part in parts[1:]:
            if part and 'PEPTIDE' in part and ':R' in part:
                connection_part = part
                break
        
        # Extract sequence from PEPTIDE{...}
        peptide_match = re.search(r'PEPTIDE\d+\{([^}]+)\}', sequence_part)
        
        if not peptide_match:
            return {
                'peptide_type': 'linear',
                'sequence': '',
                'monomers': [],
                'connections': [],
                'raw_helm': helm_sequence
            }
        
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
        """
        Parse monomer sequence into list of symbols.
        
        Handles:
            - Simple amino acids: A, G, C, etc.
            - Complex monomers: [Nle], [dF], [X2159], etc.
        
        Args:
            sequence: Dot-separated monomer string
            
        Returns:
            List of monomer symbols (without brackets)
        """
        if not sequence:
            return []
        
        monomers = []
        for monomer in sequence.split('.'):
            # Remove brackets if present
            if monomer.startswith('[') and monomer.endswith(']'):
                monomer = monomer[1:-1]
            monomers.append(monomer)
        
        return monomers
    
    def _parse_connections(self, connection_part: str) -> List[Dict]:
        """
        Parse connection information from HELM string.
        
        Connection format: PEPTIDE1,PEPTIDE1,pos1:Rx-pos2:Ry
        
        Args:
            connection_part: The connection section of HELM
            
        Returns:
            List of connection dictionaries with pos1, r1, pos2, r2
        """
        connections = []
        
        if not connection_part:
            return connections
        
        # Find all connection patterns
        connection_matches = re.findall(
            r'(\d+):R(\d+)-(\d+):R(\d+)',
            connection_part
        )
        
        for match in connection_matches:
            pos1, r1, pos2, r2 = match
            connections.append({
                'pos1': int(pos1),
                'r1': int(r1),
                'pos2': int(pos2),
                'r2': int(r2)
            })
        
        return connections
    
    def _determine_peptide_type(
        self,
        seq_length: int,
        connections: List[Dict]
    ) -> str:
        """
        Determine peptide topology type.
        
        Types:
            - linear: No connections
            - cyclic: Head-to-tail connection (pos 1 to N)
            - q_type: Other internal connections
        
        Args:
            seq_length: Length of the sequence
            connections: List of connections
            
        Returns:
            Peptide type string
        """
        if not connections:
            return 'linear'
        
        cyclic_connections = 0
        q_connections = 0
        
        for conn in connections:
            pos1, pos2 = conn['pos1'], conn['pos2']
            
            # Head-to-tail connection
            if (pos1 == 1 and pos2 == seq_length) or (pos1 == seq_length and pos2 == 1):
                cyclic_connections += 1
            else:
                q_connections += 1
        
        if cyclic_connections > 0 and q_connections == 0:
            return 'cyclic'
        elif q_connections > 0:
            return 'q_type'
        else:
            return 'linear'
    
    def get_ring_bond_matrix(
        self,
        helm_sequence: str
    ) -> Optional[Tuple[List[List[int]], List[str]]]:
        """
        Get ring bond information as a matrix.
        
        Returns upper triangular matrix where:
            - 0: No bond
            - 1: R3-R3
            - 2: R1-R2
            - 3: R1-R3
            - 4: R3-R2
        
        Args:
            helm_sequence: HELM string
            
        Returns:
            Tuple of (bond_matrix, monomer_list) or None if parsing fails
        """
        parsed = self.parse_helm_sequence(helm_sequence)
        monomers = parsed['monomers']
        connections = parsed['connections']
        
        if not monomers:
            return None
        
        n = len(monomers)
        bond_matrix = [[0] * n for _ in range(n)]
        
        for conn in connections:
            pos1, pos2 = conn['pos1'] - 1, conn['pos2'] - 1  # Convert to 0-indexed
            r1, r2 = conn['r1'], conn['r2']
            
            if 0 <= pos1 < n and 0 <= pos2 < n:
                # Determine bond type
                r_link = f"R{r1}R{r2}"
                
                # Ensure upper triangular
                if pos1 > pos2:
                    pos1, pos2 = pos2, pos1
                    r_link = f"R{r2}R{r1}"
                
                if r_link in self.BOND_TYPES:
                    bond_type = self.BOND_TYPES.index(r_link) + 1
                    bond_matrix[pos1][pos2] = bond_type
        
        return bond_matrix, monomers
    
    def extract_ring_info(self, helm_sequence: str) -> Optional[Dict]:
        """
        Extract ring bond information for training.
        
        Args:
            helm_sequence: HELM string
            
        Returns:
            Dictionary with 'sequence_length', 'bond_array' (flattened upper triangular)
        """
        result = self.get_ring_bond_matrix(helm_sequence)
        if result is None:
            return None
        
        bond_matrix, monomers = result
        n = len(monomers)
        
        # Flatten to upper triangular array
        bond_array = []
        for i in range(n):
            for j in range(i + 1, n):
                bond_array.append(bond_matrix[i][j])
        
        return {
            'sequence_length': n,
            'bond_array': bond_array,
            'num_pairs': len(bond_array)
        }
    
    def build_helm_string(
        self,
        monomers: List[str],
        connections: Optional[List[Dict]] = None
    ) -> str:
        """
        Build a HELM string from components.
        
        Args:
            monomers: List of monomer symbols
            connections: Optional list of connections
            
        Returns:
            HELM string
        """
        # Build sequence part
        sequence = '.'.join(monomers)
        sequence_part = f"PEPTIDE1{{{sequence}}}"
        
        # Build connection part
        if connections:
            conn_strings = []
            for conn in connections:
                conn_str = f"PEPTIDE1,PEPTIDE1,{conn['pos1']}:R{conn['r1']}-{conn['pos2']}:R{conn['r2']}"
                conn_strings.append(conn_str)
            connection_part = '|'.join(conn_strings)
            return f"{sequence_part}${connection_part}$$$"
        
        return f"{sequence_part}$$$$"


def test_analyzer():
    """Test the topology analyzer."""
    analyzer = HELMTopologyAnalyzer()
    
    test_cases = [
        "PEPTIDE1{[X2].[Nle].G.W.[Nle].D.F.[am]}$$$$",
        "PEPTIDE1{[X2159].[dF].C.F.W.[Lys(Boc)].[dalloT].[dC].T}$PEPTIDE1,PEPTIDE1,8:R3-3:R3$$$",
        "PEPTIDE1{A.C.G.C.K}$PEPTIDE1,PEPTIDE1,2:R3-4:R3$$$",
    ]
    
    for helm in test_cases:
        result = analyzer.parse_helm_sequence(helm)
        print(f"\nHELM: {helm}")
        print(f"Type: {result['peptide_type']}")
        print(f"Monomers: {result['monomers']}")
        print(f"Connections: {result['connections']}")


if __name__ == "__main__":
    test_analyzer()
