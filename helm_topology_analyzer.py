import re
from typing import Dict, List, Tuple, Optional

class HELMTopologyAnalyzer:
    def __init__(self):
        pass
    
    def parse_helm_sequence(self, helm_sequence: str) -> Dict:
        """解析HELM序列，提取拓扑信息"""
        if '$' in helm_sequence:
            parts = helm_sequence.split('$')
            sequence_part = parts[0] #eg:sequence_part = PEPTIDE1{C.G.C.R.K}
            connection_part = ""
            for part in parts[1:]:
                if part and 'PEPTIDE' in part and ':R' in part:
                    connection_part = part #eg:connection_part = PEPTIDE1,PEPTIDE1,6:R3-1:R3
                    break
        else:
            sequence_part = helm_sequence
            connection_part = ""
        
        peptide_match = re.search(r'PEPTIDE\d+\{([^}]+)\}', sequence_part) 

        # 默认不正常情况都是linear
        if not peptide_match:
            return {
                'peptide_type': 'linear',
                'sequence': '',
                'connections': [],
                'raw_helm': helm_sequence
            }
        
        sequence = peptide_match.group(1)
        
        connections = self._parse_connections(connection_part)
        
        peptide_type = self._determine_peptide_type(sequence, connections)
        
        return {
            'peptide_type': peptide_type,
            'sequence': sequence, # eg: [X2].[Nle].G.W.[Nle].D.F.[am]
            'connections': connections, # eg:[{'pos1': 8, 'r1': 3, 'pos2': 3, 'r2': 3}]
            'raw_helm': helm_sequence
        }
    
    def _parse_connections(self, connection_part: str) -> List[Dict]:
        """解析连接信息"""
        connections = []
        
        if not connection_part:
            return connections
        
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
    
    def _determine_peptide_type(self, sequence: str, connections: List[Dict]) -> str:
        """判断肽链类型"""
        if not connections:
            return 'linear'
        
        amino_acids = sequence.split('.')
        seq_length = len(amino_acids)
        
        cyclic_connections = 0
        q_connections = 0
        
        for conn in connections:
            pos1, pos2 = conn['pos1'], conn['pos2']
            
            # 判断是否为头尾环化 (1 to N)
            if (pos1 == 1 and pos2 == seq_length) or (pos1 == seq_length and pos2 == 1):
                cyclic_connections += 1
            else:
                # 其他连接认为是Q型
                q_connections += 1
        
        if cyclic_connections > 0 and q_connections == 0:
            return 'cyclic'
        elif q_connections > 0:
            return 'q_type'
        else:
            return 'linear'
    
    def extract_connection_matrix(self, sequence: str, connections: List[Dict]) -> Optional[List[Dict]]:
        """提取连接矩阵，保留R基团信息"""
        if not connections:
            return None
        
        amino_acids = sequence.split('.')
        seq_length = len(amino_acids)
        
        connection_pairs = []
        for conn in connections:
            pos1 = conn['pos1'] - 1
            pos2 = conn['pos2'] - 1
            
            if 0 <= pos1 < seq_length and 0 <= pos2 < seq_length:
                connection_pairs.append({
                    'pos1': pos1,
                    'pos2': pos2,
                    'r1': conn['r1'],
                    'r2': conn['r2']
                })
        
        return connection_pairs
    
    def process_batch(self, helm_sequences: List[str]) -> List[Dict]:
        """批量处理HELM序列"""
        results = []
        
        for helm_seq in helm_sequences:
            result = self.parse_helm_sequence(helm_seq)
            results.append(result)
        
        return results
    
    def get_position_encoding_info(self, helm_analysis: Dict) -> Dict:
        """获取位置编码所需信息"""
        peptide_type = helm_analysis['peptide_type']
        connections = helm_analysis.get('connections', [])
        sequence = helm_analysis.get('sequence', '')
        
        amino_acids = sequence.split('.')
        seq_length = len(amino_acids)
        
        connection_info = None
        if connections:
            connection_info = self.extract_connection_matrix(sequence, connections)
        
        return {
            'peptide_type': peptide_type,
            'seq_length': seq_length,
            'connection_info': connection_info
        }

def test_analyzer():
    """测试分析器"""
    analyzer = HELMTopologyAnalyzer()
    
    # 测试用例
    test_cases = [
        # 线性肽
        "PEPTIDE1{[X2].[Nle].G.W.[Nle].D.F.[am]}$$$$",
        # 环形肽 (头尾连接)
        "PEPTIDE1{[X2159].[dF].C.F.W.[Lys(Boc)].[dalloT].[dC].T}$PEPTIDE1,PEPTIDE1,8:R3-3:R3$$$",
        # Q型肽 (侧链连接)
        "PEPTIDE1{A.C.G.C.K}$PEPTIDE1,PEPTIDE1,2:R1-4:R2$$$",
    ]
    
    for helm_seq in test_cases:
        result = analyzer.parse_helm_sequence(helm_seq)
        print(f"\nHELM: {helm_seq}")
        print(f"Type: {result['peptide_type']}")
        print(f"Sequence: {result['sequence']}")
        print(f"Connections: {result['connections']}")
        
        pos_info = analyzer.get_position_encoding_info(result)
        print(f"Position encoding info: {pos_info}")

if __name__ == "__main__":
    test_analyzer()
