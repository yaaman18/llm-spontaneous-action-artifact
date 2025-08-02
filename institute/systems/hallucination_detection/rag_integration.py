"""
RAG (Retrieval-Augmented Generation) 統合システム
外部知識ベースとの実時間連携による幻覚防止
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: FAISS or sentence-transformers not installed. Some features may not work.")

@dataclass
class KnowledgeSource:
    source_id: str
    source_type: str  # "paper", "database", "expert_memory", "web"
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    reliability_score: float = 0.8
    last_updated: datetime = datetime.now()

@dataclass 
class RetrievalResult:
    query: str
    retrieved_sources: List[KnowledgeSource]
    relevance_scores: List[float]
    confidence: float
    timestamp: datetime

class EmbeddingManager:
    """ベクトル埋め込み管理システム"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.model = None
            self.dimension = 384  # デフォルト次元
        
        # FAISS インデックス
        self.index = None
        self.source_metadata: List[KnowledgeSource] = []
        
    def initialize_index(self, sources: List[KnowledgeSource]):
        """FAISSインデックスを初期化"""
        if not self.model:
            print("Warning: Embedding model not available, skipping index creation")
            return
            
        try:
            # エンベディング生成
            texts = [source.content for source in sources]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # FAISSインデックス構築
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product index
            self.index.add(embeddings.astype('float32'))
            
            # メタデータ保存
            for i, source in enumerate(sources):
                source.embedding = embeddings[i]
                self.source_metadata.append(source)
                
            print(f"Initialized FAISS index with {len(sources)} sources")
            
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
    
    async def search_similar(self, query: str, k: int = 5) -> List[Tuple[KnowledgeSource, float]]:
        """類似度検索"""
        if not self.model or not self.index:
            print("Warning: Search not available - model or index not initialized")
            return []
        
        try:
            # クエリエンベディング
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # 検索実行
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # 結果構築
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.source_metadata):
                    source = self.source_metadata[idx]
                    results.append((source, float(score)))
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

class KnowledgeBaseManager:
    """知識ベース管理システム"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_manager = EmbeddingManager()
        self.knowledge_sources: Dict[str, KnowledgeSource] = {}
        
        # 各種データソース
        self.paper_db_path = self.data_dir / "papers"
        self.expert_memory_path = self.data_dir / "expert_memories"
        self.web_cache_path = self.data_dir / "web_cache"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        for path in [self.paper_db_path, self.expert_memory_path, self.web_cache_path]:
            path.mkdir(exist_ok=True)
    
    async def initialize(self):
        """知識ベースを初期化"""
        print("Initializing knowledge base...")
        
        # 各ソースから知識を読み込み
        await self._load_papers()
        await self._load_expert_memories()
        await self._load_web_cache()
        
        # エンベディングインデックス構築
        sources = list(self.knowledge_sources.values())
        if sources:
            self.embedding_manager.initialize_index(sources)
        
        print(f"Knowledge base initialized with {len(sources)} sources")
    
    async def _load_papers(self):
        """論文データベースから読み込み"""
        papers_path = self.paper_db_path
        if not papers_path.exists():
            return
        
        # マークダウンファイルを読み込み
        for md_file in papers_path.glob("**/*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # メタデータ抽出
                metadata = self._extract_paper_metadata(content)
                
                source = KnowledgeSource(
                    source_id=f"paper_{md_file.stem}",
                    source_type="paper",
                    title=metadata.get("title", md_file.stem),
                    content=content,
                    metadata=metadata,
                    reliability_score=0.9  # 論文は高信頼度
                )
                
                self.knowledge_sources[source.source_id] = source
                
            except Exception as e:
                print(f"Error loading paper {md_file}: {e}")
    
    async def _load_expert_memories(self):
        """エキスパートメモリから読み込み"""
        memory_base = Path("/Users/yamaguchimitsuyuki/omoikane-lab/memory")
        
        if not memory_base.exists():
            return
        
        # エージェントの記憶を読み込み
        for agent_dir in memory_base.glob("agents/*"):
            if not agent_dir.is_dir():
                continue
                
            agent_name = agent_dir.name
            
            # 各記憶ファイルを読み込み
            for memory_file in agent_dir.glob("*.md"):
                try:
                    content = memory_file.read_text(encoding='utf-8')
                    
                    source = KnowledgeSource(
                        source_id=f"memory_{agent_name}_{memory_file.stem}",
                        source_type="expert_memory",
                        title=f"{agent_name} - {memory_file.stem}",
                        content=content,
                        metadata={
                            "agent": agent_name,
                            "memory_type": memory_file.stem
                        },
                        reliability_score=0.8  # エキスパート記憶は高信頼度
                    )
                    
                    self.knowledge_sources[source.source_id] = source
                    
                except Exception as e:
                    print(f"Error loading memory {memory_file}: {e}")
    
    async def _load_web_cache(self):
        """Webキャッシュから読み込み"""
        # 実装: Webから取得した情報のキャッシュ
        # 現在はプレースホルダー
        pass
    
    def _extract_paper_metadata(self, content: str) -> Dict[str, Any]:
        """論文のメタデータを抽出"""
        metadata = {}
        
        # YAMLフロントマター抽出
        if content.startswith("---"):
            try:
                yaml_end = content.find("---", 3)
                if yaml_end > 0:
                    yaml_content = content[3:yaml_end]
                    # 簡易YAML解析
                    for line in yaml_content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[key.strip()] = value.strip()
            except:
                pass
        
        # タイトル抽出
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                metadata['title'] = line[2:].strip()
                break
        
        return metadata
    
    async def retrieve_relevant_knowledge(self, 
                                        query: str, 
                                        k: int = 5,
                                        source_types: List[str] = None) -> RetrievalResult:
        """関連知識を検索"""
        
        # 類似度検索
        similar_sources = await self.embedding_manager.search_similar(query, k * 2)
        
        # ソースタイプフィルタリング
        if source_types:
            similar_sources = [
                (source, score) for source, score in similar_sources
                if source.source_type in source_types
            ]
        
        # 上位k件を選択
        top_sources = similar_sources[:k]
        
        if not top_sources:
            return RetrievalResult(
                query=query,
                retrieved_sources=[],
                relevance_scores=[],
                confidence=0.0,
                timestamp=datetime.now()
            )
        
        # 結果構築
        sources = [source for source, _ in top_sources]
        scores = [score for _, score in top_sources]
        
        # 信頼度計算
        confidence = self._calculate_retrieval_confidence(scores, sources)
        
        return RetrievalResult(
            query=query,
            retrieved_sources=sources,
            relevance_scores=scores,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def _calculate_retrieval_confidence(self, 
                                      scores: List[float], 
                                      sources: List[KnowledgeSource]) -> float:
        """検索結果の信頼度を計算"""
        if not scores:
            return 0.0
        
        # スコアベースの信頼度
        avg_score = np.mean(scores)
        score_confidence = min(1.0, avg_score / 0.8)  # 0.8を最大とする
        
        # ソース信頼度
        source_reliability = np.mean([s.reliability_score for s in sources])
        
        # 多様性ボーナス
        source_types = set(s.source_type for s in sources)
        diversity_bonus = min(1.0, len(source_types) / 3.0)
        
        # 総合信頼度
        overall_confidence = (
            score_confidence * 0.5 +
            source_reliability * 0.3 +
            diversity_bonus * 0.2
        )
        
        return float(np.clip(overall_confidence, 0.0, 1.0))

class RAGIntegration:
    """RAG統合システム"""
    
    def __init__(self, knowledge_base_path: Path):
        self.kb_manager = KnowledgeBaseManager(knowledge_base_path)
        self.retrieval_history: List[RetrievalResult] = []
        
    async def initialize(self):
        """システム初期化"""
        await self.kb_manager.initialize()
        print("RAG Integration system initialized")
    
    async def verify_statement_with_sources(self, 
                                          statement: str, 
                                          context: str = None) -> Dict[str, Any]:
        """文を外部ソースで検証"""
        
        # 関連知識検索
        retrieval_result = await self.kb_manager.retrieve_relevant_knowledge(
            statement, k=5
        )
        
        # 履歴記録
        self.retrieval_history.append(retrieval_result)
        
        # 検証分析
        verification_analysis = await self._analyze_verification(
            statement, retrieval_result
        )
        
        return {
            "statement": statement,
            "retrieval_result": retrieval_result,
            "verification": verification_analysis,
            "timestamp": datetime.now()
        }
    
    async def _analyze_verification(self, 
                                  statement: str, 
                                  retrieval: RetrievalResult) -> Dict[str, Any]:
        """検証分析を実行"""
        
        if not retrieval.retrieved_sources:
            return {
                "support_level": "no_evidence",
                "confidence": 0.0,
                "findings": "No relevant sources found",
                "contradictions": [],
                "supporting_evidence": []
            }
        
        # 支持度分析
        supporting_evidence = []
        contradictions = []
        
        for source in retrieval.retrieved_sources:
            # 簡易分析（実際にはより詳細な意味解析が必要）
            content_lower = source.content.lower()
            statement_lower = statement.lower()
            
            # キーワード一致度
            statement_words = set(statement_lower.split())
            content_words = set(content_lower.split())
            overlap = len(statement_words & content_words)
            
            if overlap >= 3:  # 簡易閾値
                supporting_evidence.append({
                    "source": source.title,
                    "relevance": overlap / len(statement_words),
                    "excerpt": source.content[:200] + "..."
                })
        
        # 支持レベル判定
        if len(supporting_evidence) >= 2:
            support_level = "strong_support"
            confidence = 0.8
        elif len(supporting_evidence) == 1:
            support_level = "moderate_support"
            confidence = 0.6
        else:
            support_level = "weak_support"
            confidence = 0.3
        
        return {
            "support_level": support_level,
            "confidence": confidence,
            "findings": f"Found {len(supporting_evidence)} supporting sources",
            "contradictions": contradictions,
            "supporting_evidence": supporting_evidence
        }
    
    async def get_authoritative_sources(self, topic: str) -> List[KnowledgeSource]:
        """権威ある情報源を取得"""
        
        retrieval = await self.kb_manager.retrieve_relevant_knowledge(
            topic, k=10, source_types=["paper", "expert_memory"]
        )
        
        # 信頼度でソート
        authoritative = sorted(
            retrieval.retrieved_sources,
            key=lambda s: s.reliability_score,
            reverse=True
        )
        
        return authoritative[:5]
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """検索統計を取得"""
        if not self.retrieval_history:
            return {"total_retrievals": 0}
        
        total = len(self.retrieval_history)
        avg_confidence = np.mean([r.confidence for r in self.retrieval_history])
        
        source_type_counts = {}
        for result in self.retrieval_history:
            for source in result.retrieved_sources:
                source_type_counts[source.source_type] = \
                    source_type_counts.get(source.source_type, 0) + 1
        
        return {
            "total_retrievals": total,
            "average_confidence": float(avg_confidence),
            "source_type_distribution": source_type_counts
        }

# 使用例
async def main():
    """RAGシステムテスト"""
    
    # システム初期化
    kb_path = Path("/Users/yamaguchimitsuyuki/omoikane-lab/institute/knowledge_base")
    rag = RAGIntegration(kb_path)
    await rag.initialize()
    
    # テスト文の検証
    test_statement = "統合情報理論では、意識はΦ値によって定量化される"
    
    result = await rag.verify_statement_with_sources(test_statement)
    
    print(f"Statement: {result['statement']}")
    print(f"Support Level: {result['verification']['support_level']}")
    print(f"Confidence: {result['verification']['confidence']:.2f}")
    print(f"Supporting Evidence: {len(result['verification']['supporting_evidence'])}")

if __name__ == "__main__":
    asyncio.run(main())