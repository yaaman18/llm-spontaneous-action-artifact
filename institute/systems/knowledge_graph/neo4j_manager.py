"""
Neo4j 知識グラフ管理システム
研究所の知識を関係性グラフとして管理・検索
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid

try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError:
    print("Warning: neo4j driver not installed. Install with: pip install neo4j")
    GraphDatabase = None
    AsyncGraphDatabase = None

class NodeType(Enum):
    CONCEPT = "Concept"
    RESEARCHER = "Researcher"
    PAPER = "Paper"
    THEORY = "Theory"
    METHOD = "Method"
    FINDING = "Finding"
    DOMAIN = "Domain"
    STATEMENT = "Statement"
    VERIFICATION = "Verification"
    CONTRADICTION = "Contradiction"

class RelationType(Enum):
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    AUTHORED_BY = "AUTHORED_BY"
    BELONGS_TO = "BELONGS_TO"
    DEVELOPS = "DEVELOPS"
    APPLIES = "APPLIES"
    CITES = "CITES"
    VERIFIED_BY = "VERIFIED_BY"
    BUILDS_ON = "BUILDS_ON"
    SPECIALIZES = "SPECIALIZES"

@dataclass
class KnowledgeNode:
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0
    source: str = "system"

@dataclass
class KnowledgeRelation:
    id: str
    type: RelationType
    from_node_id: str
    to_node_id: str
    properties: Dict[str, Any]
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime
    source: str = "system"

@dataclass
class GraphQuery:
    query: str
    parameters: Dict[str, Any]
    result_limit: int = 100

class Neo4jKnowledgeGraph:
    """Neo4j 知識グラフマネージャー"""
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", 
                 password: str = "password",
                 database: str = "omoikane"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.is_connected = False
        
        # インデックス管理
        self.node_indexes = {
            NodeType.CONCEPT: ["name", "domain"],
            NodeType.RESEARCHER: ["name", "expertise"],
            NodeType.PAPER: ["title", "authors"],
            NodeType.THEORY: ["name", "domain"],
            NodeType.STATEMENT: ["content", "domain"]
        }
        
        # 制約管理
        self.constraints = {
            NodeType.CONCEPT: ["name"],
            NodeType.RESEARCHER: ["name"],
            NodeType.PAPER: ["title"],
            NodeType.THEORY: ["name"]
        }
    
    async def initialize(self):
        """データベース接続とスキーマ初期化"""
        try:
            if AsyncGraphDatabase:
                self.driver = AsyncGraphDatabase.driver(
                    self.uri, 
                    auth=(self.user, self.password)
                )
                
                # 接続テスト
                await self._test_connection()
                
                # スキーマ設定
                await self._setup_schema()
                
                self.is_connected = True
                print("Neo4j knowledge graph initialized successfully")
            else:
                print("Warning: Neo4j driver not available. Using mock implementation.")
                self.is_connected = False
                
        except Exception as e:
            print(f"Failed to initialize Neo4j: {e}")
            self.is_connected = False
    
    async def _test_connection(self):
        """接続テスト"""
        if not self.driver:
            return
            
        async with self.driver.session(database=self.database) as session:
            result = await session.run("RETURN 1 as test")
            await result.single()
    
    async def _setup_schema(self):
        """スキーマとインデックスを設定"""
        if not self.driver:
            return
            
        async with self.driver.session(database=self.database) as session:
            
            # 制約作成
            for node_type, properties in self.constraints.items():
                for prop in properties:
                    constraint_query = f"""
                    CREATE CONSTRAINT {node_type.value.lower()}_{prop}_unique IF NOT EXISTS
                    FOR (n:{node_type.value}) REQUIRE n.{prop} IS UNIQUE
                    """
                    try:
                        await session.run(constraint_query)
                    except ClientError as e:
                        if "equivalent constraint already exists" not in str(e):
                            print(f"Warning: Failed to create constraint: {e}")
            
            # インデックス作成
            for node_type, properties in self.node_indexes.items():
                for prop in properties:
                    index_query = f"""
                    CREATE INDEX {node_type.value.lower()}_{prop}_index IF NOT EXISTS
                    FOR (n:{node_type.value}) ON (n.{prop})
                    """
                    try:
                        await session.run(index_query)
                    except ClientError as e:
                        if "equivalent index already exists" not in str(e):
                            print(f"Warning: Failed to create index: {e}")
    
    async def add_node(self, node: KnowledgeNode) -> bool:
        """ノードを追加"""
        if not self.is_connected:
            print(f"Mock: Adding node {node.name} of type {node.type.value}")
            return True
            
        try:
            async with self.driver.session(database=self.database) as session:
                query = f"""
                MERGE (n:{node.type.value} {{id: $id}})
                SET n.name = $name,
                    n.created_at = $created_at,
                    n.updated_at = $updated_at,
                    n.confidence = $confidence,
                    n.source = $source
                """
                
                # プロパティを動的に追加
                property_sets = []
                parameters = {
                    'id': node.id,
                    'name': node.name,
                    'created_at': node.created_at.isoformat(),
                    'updated_at': node.updated_at.isoformat(),
                    'confidence': node.confidence,
                    'source': node.source
                }
                
                for key, value in node.properties.items():
                    property_sets.append(f"n.{key} = ${key}")
                    parameters[key] = value
                
                if property_sets:
                    query += ",\n    " + ",\n    ".join(property_sets)
                
                await session.run(query, parameters)
                return True
                
        except Exception as e:
            print(f"Error adding node: {e}")
            return False
    
    async def add_relation(self, relation: KnowledgeRelation) -> bool:
        """リレーションを追加"""
        if not self.is_connected:
            print(f"Mock: Adding relation {relation.type.value} from {relation.from_node_id} to {relation.to_node_id}")
            return True
            
        try:
            async with self.driver.session(database=self.database) as session:
                query = f"""
                MATCH (from_node {{id: $from_id}}), (to_node {{id: $to_id}})
                MERGE (from_node)-[r:{relation.type.value} {{id: $relation_id}}]->(to_node)
                SET r.weight = $weight,
                    r.confidence = $confidence,
                    r.created_at = $created_at,
                    r.source = $source
                """
                
                parameters = {
                    'from_id': relation.from_node_id,
                    'to_id': relation.to_node_id,
                    'relation_id': relation.id,
                    'weight': relation.weight,
                    'confidence': relation.confidence,
                    'created_at': relation.created_at.isoformat(),
                    'source': relation.source
                }
                
                # プロパティを動的に追加
                property_sets = []
                for key, value in relation.properties.items():
                    property_sets.append(f"r.{key} = ${key}")
                    parameters[key] = value
                
                if property_sets:
                    query += ",\n    " + ",\n    ".join(property_sets)
                
                await session.run(query, parameters)
                return True
                
        except Exception as e:
            print(f"Error adding relation: {e}")
            return False
    
    async def find_nodes(self, 
                        node_type: NodeType = None,
                        properties: Dict[str, Any] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """ノードを検索"""
        if not self.is_connected:
            print(f"Mock: Finding nodes of type {node_type}")
            return [{"id": "mock_node", "name": "Mock Node", "type": node_type.value if node_type else "Unknown"}]
        
        try:
            async with self.driver.session(database=self.database) as session:
                
                # クエリ構築
                if node_type:
                    match_clause = f"MATCH (n:{node_type.value})"
                else:
                    match_clause = "MATCH (n)"
                
                where_clauses = []
                parameters = {}
                
                if properties:
                    for key, value in properties.items():
                        param_name = f"{key}_param"
                        where_clauses.append(f"n.{key} = ${param_name}")
                        parameters[param_name] = value
                
                where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
                
                query = f"""
                {match_clause}
                {where_clause}
                RETURN n
                LIMIT {limit}
                """
                
                result = await session.run(query, parameters)
                nodes = []
                
                async for record in result:
                    node_data = dict(record['n'])
                    nodes.append(node_data)
                
                return nodes
                
        except Exception as e:
            print(f"Error finding nodes: {e}")
            return []
    
    async def find_related_concepts(self, 
                                  concept_name: str, 
                                  max_depth: int = 2,
                                  relation_types: List[RelationType] = None) -> Dict[str, Any]:
        """関連概念を検索"""
        if not self.is_connected:
            return {
                "center_concept": concept_name,
                "related_concepts": [
                    {"name": "Mock Related Concept 1", "relation": "RELATED_TO", "distance": 1},
                    {"name": "Mock Related Concept 2", "relation": "SUPPORTS", "distance": 2}
                ]
            }
        
        try:
            async with self.driver.session(database=self.database) as session:
                
                # リレーションタイプフィルター
                relation_filter = ""
                if relation_types:
                    relation_names = [rt.value for rt in relation_types]
                    relation_filter = f"WHERE type(r) IN {relation_names}"
                
                query = f"""
                MATCH path = (center:Concept {{name: $concept_name}})-[r*1..{max_depth}]-(related)
                {relation_filter}
                RETURN 
                    related.name as name,
                    type(last(relationships(path))) as relation_type,
                    length(path) as distance,
                    related.domain as domain,
                    avg([rel in relationships(path) | rel.confidence]) as avg_confidence
                ORDER BY distance, avg_confidence DESC
                LIMIT 20
                """
                
                result = await session.run(query, {'concept_name': concept_name})
                related_concepts = []
                
                async for record in result:
                    related_concepts.append({
                        'name': record['name'],
                        'relation': record['relation_type'],
                        'distance': record['distance'],
                        'domain': record['domain'],
                        'confidence': record['avg_confidence']
                    })
                
                return {
                    'center_concept': concept_name,
                    'related_concepts': related_concepts
                }
                
        except Exception as e:
            print(f"Error finding related concepts: {e}")
            return {'center_concept': concept_name, 'related_concepts': []}
    
    async def detect_contradictions(self, domain: str = None) -> List[Dict[str, Any]]:
        """矛盾を検出"""
        if not self.is_connected:
            return [
                {
                    "concept1": "Mock Concept A",
                    "concept2": "Mock Concept B", 
                    "contradiction_type": "CONTRADICTS",
                    "confidence": 0.8
                }
            ]
        
        try:
            async with self.driver.session(database=self.database) as session:
                
                domain_filter = f"WHERE c1.domain = '{domain}' AND c2.domain = '{domain}'" if domain else ""
                
                query = f"""
                MATCH (c1:Concept)-[r:CONTRADICTS]->(c2:Concept)
                {domain_filter}
                RETURN 
                    c1.name as concept1,
                    c2.name as concept2,
                    r.type as contradiction_type,
                    r.confidence as confidence,
                    r.evidence as evidence
                ORDER BY r.confidence DESC
                LIMIT 10
                """
                
                result = await session.run(query)
                contradictions = []
                
                async for record in result:
                    contradictions.append({
                        'concept1': record['concept1'],
                        'concept2': record['concept2'],
                        'contradiction_type': record['contradiction_type'],
                        'confidence': record['confidence'],
                        'evidence': record['evidence']
                    })
                
                return contradictions
                
        except Exception as e:
            print(f"Error detecting contradictions: {e}")
            return []
    
    async def find_knowledge_gaps(self, domain: str) -> List[Dict[str, Any]]:
        """知識ギャップを特定"""
        if not self.is_connected:
            return [
                {
                    "gap_type": "missing_connection",
                    "concept1": "Mock Concept X",
                    "concept2": "Mock Concept Y",
                    "suggested_relation": "RELATED_TO"
                }
            ]
        
        try:
            async with self.driver.session(database=self.database) as session:
                
                # 孤立した概念を検出
                isolated_query = f"""
                MATCH (c:Concept {{domain: $domain}})
                WHERE NOT (c)-[]-()
                RETURN c.name as isolated_concept
                LIMIT 5
                """
                
                result = await session.run(isolated_query, {'domain': domain})
                gaps = []
                
                async for record in result:
                    gaps.append({
                        'gap_type': 'isolated_concept',
                        'concept': record['isolated_concept'],
                        'suggestion': 'このコンセプトに関連を追加してください'
                    })
                
                # 弱い結合を検出
                weak_connection_query = f"""
                MATCH (c1:Concept {{domain: $domain}})-[r]-(c2:Concept {{domain: $domain}})
                WHERE r.confidence < 0.5
                RETURN 
                    c1.name as concept1,
                    c2.name as concept2,
                    type(r) as relation_type,
                    r.confidence as confidence
                ORDER BY r.confidence ASC
                LIMIT 5
                """
                
                result = await session.run(weak_connection_query, {'domain': domain})
                
                async for record in result:
                    gaps.append({
                        'gap_type': 'weak_connection',
                        'concept1': record['concept1'],
                        'concept2': record['concept2'],
                        'relation': record['relation_type'],
                        'confidence': record['confidence'],
                        'suggestion': 'この関係の信頼度を向上させてください'
                    })
                
                return gaps
                
        except Exception as e:
            print(f"Error finding knowledge gaps: {e}")
            return []
    
    async def close(self):
        """接続を閉じる"""
        if self.driver:
            await self.driver.close()

class KnowledgeGraphBuilder:
    """知識グラフ構築ヘルパー"""
    
    def __init__(self, graph_manager: Neo4jKnowledgeGraph):
        self.graph_manager = graph_manager
    
    async def build_from_papers(self, papers_directory: str):
        """論文から知識グラフを構築"""
        
        # 論文ディレクトリをスキャン
        from pathlib import Path
        papers_path = Path(papers_directory)
        
        if not papers_path.exists():
            print(f"Papers directory not found: {papers_directory}")
            return
        
        for paper_file in papers_path.glob("**/*.md"):
            await self._process_paper_file(paper_file)
        
        print(f"Knowledge graph built from papers in {papers_directory}")
    
    async def _process_paper_file(self, paper_file: Path):
        """論文ファイルを処理してグラフに追加"""
        
        try:
            content = paper_file.read_text(encoding='utf-8')
            
            # メタデータ抽出
            metadata = self._extract_paper_metadata(content)
            
            # 論文ノード作成
            paper_node = KnowledgeNode(
                id=str(uuid.uuid4()),
                type=NodeType.PAPER,
                name=metadata.get('title', paper_file.stem),
                properties={
                    'file_path': str(paper_file),
                    'authors': metadata.get('authors', []),
                    'year': metadata.get('year'),
                    'domain': metadata.get('domain', 'unknown')
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            await self.graph_manager.add_node(paper_node)
            
            # 概念抽出とノード作成
            concepts = self._extract_concepts(content)
            for concept in concepts:
                concept_node = KnowledgeNode(
                    id=str(uuid.uuid4()),
                    type=NodeType.CONCEPT,
                    name=concept['name'],
                    properties={
                        'domain': concept.get('domain', metadata.get('domain', 'unknown')),
                        'frequency': concept.get('frequency', 1)
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                await self.graph_manager.add_node(concept_node)
                
                # 論文-概念関係を追加
                relation = KnowledgeRelation(
                    id=str(uuid.uuid4()),
                    type=RelationType.RELATED_TO,
                    from_node_id=paper_node.id,
                    to_node_id=concept_node.id,
                    properties={'context': 'paper_mentions'},
                    weight=concept.get('frequency', 1),
                    created_at=datetime.now()
                )
                
                await self.graph_manager.add_relation(relation)
                
        except Exception as e:
            print(f"Error processing paper {paper_file}: {e}")
    
    def _extract_paper_metadata(self, content: str) -> Dict[str, Any]:
        """論文メタデータを抽出"""
        metadata = {}
        
        # YAMLフロントマター解析（簡易版）
        if content.startswith('---'):
            try:
                yaml_end = content.find('---', 3)
                if yaml_end > 0:
                    yaml_content = content[3:yaml_end]
                    for line in yaml_content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
            except:
                pass
        
        return metadata
    
    def _extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """概念を抽出"""
        
        # 重要な概念キーワード（実際にはより高度なNLPが必要）
        concept_keywords = [
            'consciousness', 'phi', 'integrated information',
            'global workspace', 'phenomenology', 'qualia',
            'intentionality', 'temporal consciousness'
        ]
        
        concepts = []
        content_lower = content.lower()
        
        for keyword in concept_keywords:
            if keyword in content_lower:
                frequency = content_lower.count(keyword)
                concepts.append({
                    'name': keyword,
                    'frequency': frequency,
                    'domain': self._infer_domain(keyword)
                })
        
        return concepts
    
    def _infer_domain(self, concept: str) -> str:
        """概念からドメインを推論"""
        
        domain_mappings = {
            'consciousness': 'consciousness_studies',
            'phi': 'consciousness_studies', 
            'phenomenology': 'philosophy',
            'qualia': 'philosophy',
            'global workspace': 'cognitive_science'
        }
        
        return domain_mappings.get(concept.lower(), 'general')
    
    async def build_researcher_network(self, agent_configs_dir: str):
        """研究者ネットワークを構築"""
        
        from pathlib import Path
        import yaml
        
        configs_path = Path(agent_configs_dir)
        
        if not configs_path.exists():
            print(f"Agent configs directory not found: {agent_configs_dir}")
            return
        
        for config_file in configs_path.glob("*.yaml"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                agent_info = config.get('agent', {})
                
                # 研究者ノード作成
                researcher_node = KnowledgeNode(
                    id=str(uuid.uuid4()),
                    type=NodeType.RESEARCHER,
                    name=agent_info.get('name', config_file.stem),
                    properties={
                        'role': agent_info.get('role', ''),
                        'department': agent_info.get('department', ''),
                        'expertise': agent_info.get('expertise', {}),
                        'config_file': str(config_file)
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                await self.graph_manager.add_node(researcher_node)
                
                # 専門分野との関係追加
                expertise = agent_info.get('expertise', {})
                for domain, concepts in expertise.items():
                    if isinstance(concepts, list):
                        for concept in concepts:
                            # 概念ノード作成/取得
                            concept_node = KnowledgeNode(
                                id=str(uuid.uuid4()),
                                type=NodeType.CONCEPT,
                                name=concept,
                                properties={'domain': domain},
                                created_at=datetime.now(),
                                updated_at=datetime.now()
                            )
                            
                            await self.graph_manager.add_node(concept_node)
                            
                            # 専門関係追加
                            relation = KnowledgeRelation(
                                id=str(uuid.uuid4()),
                                type=RelationType.SPECIALIZES,
                                from_node_id=researcher_node.id,
                                to_node_id=concept_node.id,
                                properties={'expertise_level': 'primary'},
                                weight=1.0,
                                created_at=datetime.now()
                            )
                            
                            await self.graph_manager.add_relation(relation)
                
            except Exception as e:
                print(f"Error processing agent config {config_file}: {e}")

# 使用例
async def main():
    """知識グラフシステムテスト"""
    
    # Neo4j 接続
    graph_manager = Neo4jKnowledgeGraph()
    await graph_manager.initialize()
    
    # テスト用概念ノード追加
    consciousness_node = KnowledgeNode(
        id=str(uuid.uuid4()),
        type=NodeType.CONCEPT,
        name="consciousness",
        properties={'domain': 'consciousness_studies', 'definition': 'subjective experience'},
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    await graph_manager.add_node(consciousness_node)
    
    # 関連概念検索
    related = await graph_manager.find_related_concepts("consciousness")
    print(f"Related concepts: {related}")
    
    # 知識ギャップ検出
    gaps = await graph_manager.find_knowledge_gaps("consciousness_studies")
    print(f"Knowledge gaps: {gaps}")
    
    # 知識グラフ構築
    builder = KnowledgeGraphBuilder(graph_manager)
    
    # 論文から構築
    await builder.build_from_papers("/Users/yamaguchimitsuyuki/omoikane-lab/institute/tools/paper-collector")
    
    # 研究者ネットワーク構築  
    await builder.build_researcher_network("/Users/yamaguchimitsuyuki/omoikane-lab/institute/agents")
    
    await graph_manager.close()

if __name__ == "__main__":
    asyncio.run(main())