import re
from typing import Dict, List, Tuple, Optional
"""
HELM序列拓扑分析器，主要方法是parse_helm_sequence
返还例子：
HELM: PEPTIDE1{[X2].[Nle].G.W.[Nle].D.F.[am]}$$$$
Type: linear
Sequence: [X2].[Nle].G.W.[Nle].D.F.[am]
Connections: []

HELM: PEPTIDE1{[X2159].[dF].C.F.W.[Lys(Boc)].[dalloT].[dC].T}$PEPTIDE1,PEPTIDE1,8:R3-3:R3$$$
Type: q_type
Sequence: [X2159].[dF].C.F.W.[Lys(Boc)].[dalloT].[dC].T
Connections: [{'pos1': 8, 'r1': 3, 'pos2': 3, 'r2': 3}]
"""
class HELMTopologyAnalyzer:
    def __init__(self):
        pass
    
    def parse_helm_sequence(self, helm_sequence: str) -> Dict:
        """解析HELM序列，提取拓扑信息"""
        parts = helm_sequence.split('$')
        sequence_part = parts[0] #eg:sequence_part = PEPTIDE1{C.G.C.R.K}
        connection_part = ""
        for part in parts[1:]:
            if part and 'PEPTIDE' in part and ':R' in part:
                connection_part = part #eg:connection_part = PEPTIDE1,PEPTIDE1,6:R3-1:R3
                break
        
        peptide_match = re.search(r'PEPTIDE\d+\{([^}]+)\}', sequence_part) #eg: [X2].[Nle].G.W.[Nle].D.F.[am]

        # 不正常情况抛出错误
        if not peptide_match:
            print("存在sequence_part无法被正常提取")
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
            'peptide_type': peptide_type, # 'linear', 'cyclic', 'q_type'
            'sequence': sequence, # eg: [X2].[Nle].G.W.[Nle].D.F.[am]
            'connections': connections, # eg:[{'pos1': 8, 'r1': 3, 'pos2': 3, 'r2': 3}]
            'raw_helm': helm_sequence # eg: PEPTIDE1{C.G.C.R.K}$PEPTIDE1,PEPTIDE1,6:R3-1:R3$$$
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
        

if __name__ == "__main__":
    test_analyzer()
