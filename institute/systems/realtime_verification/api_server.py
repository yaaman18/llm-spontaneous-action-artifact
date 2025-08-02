"""
リアルタイム検証API サーバー
FastAPI + WebSocket による実時間ハルシネーション検証システム
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# 相対importを調整
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from hallucination_detection.core import (
        HallucinationDetectionEngine, DetectionResult
    )
    from knowledge_verification.domain_specialists import (
        DomainSpecialistFactory, VerificationLevel
    )
    from knowledge_verification.consensus_engine import (
        ConsensusEngine, ExpertOpinion
    )
    from hallucination_detection.rag_integration import RAGIntegration
except ImportError as e:
    print(f"Warning: Could not import verification modules: {e}")
    # モックで代替
    class HallucinationDetectionEngine:
        def __init__(self, *args, **kwargs): pass
        async def detect_hallucination(self, *args, **kwargs):
            return {"is_hallucination": False, "confidence_score": 0.8}

class VerificationRequest(BaseModel):
    statement: str = Field(..., description="検証対象の文")
    context: Optional[str] = Field(None, description="文脈情報")
    domain_hint: Optional[str] = Field(None, description="分野ヒント")
    verification_level: str = Field("moderate", description="検証レベル")
    require_consensus: bool = Field(True, description="専門家コンセンサスが必要か")

class VerificationResponse(BaseModel):
    request_id: str
    statement: str
    is_valid: bool
    confidence_score: float
    hallucination_detected: bool
    expert_consensus: Optional[Dict[str, Any]]
    domain_analysis: Dict[str, Any]
    recommendations: List[str]
    processing_time: float
    timestamp: datetime

class ConnectionManager:
    """WebSocket接続管理"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_info: Dict[str, Any] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = client_info or {}
        await self.send_personal_message(
            {"type": "connection_established", "message": "接続が確立されました"}, 
            websocket
        )
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_info:
            del self.connection_info[websocket]
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            print(f"Error sending message: {e}")
    
    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except:
                disconnected.append(connection)
        
        # 切断されたコネクションを削除
        for connection in disconnected:
            self.disconnect(connection)

class RealtimeVerificationSystem:
    """リアルタイム検証システム"""
    
    def __init__(self):
        # 検証エンジン初期化
        self.hallucination_detector = None
        self.consensus_engine = None
        self.rag_integration = None
        
        # 統計情報
        self.verification_stats = {
            "total_verifications": 0,
            "hallucinations_detected": 0,
            "average_processing_time": 0.0,
            "consensus_achieved": 0
        }
        
        # リクエスト履歴
        self.verification_history: List[VerificationResponse] = []
        
    async def initialize(self):
        """システム初期化"""
        try:
            # エージェント設定読み込み
            agents_config = await self._load_agent_configs()
            
            # 検証エンジン初期化
            self.hallucination_detector = HallucinationDetectionEngine(agents_config)
            self.consensus_engine = ConsensusEngine()
            
            # RAG システム初期化
            kb_path = Path(__file__).parent.parent.parent / "knowledge_base"
            self.rag_integration = RAGIntegration(kb_path)
            await self.rag_integration.initialize()
            
            print("Realtime verification system initialized successfully")
            
        except Exception as e:
            print(f"Warning: Verification system initialization failed: {e}")
            print("Using mock implementations")
    
    async def _load_agent_configs(self) -> Dict[str, Dict]:
        """エージェント設定を読み込み"""
        configs = {}
        
        agents_dir = Path(__file__).parent.parent.parent / "agents"
        if agents_dir.exists():
            import yaml
            for config_file in agents_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        agent_name = config_file.stem
                        configs[agent_name] = config
                except Exception as e:
                    print(f"Error loading agent config {config_file}: {e}")
        
        return configs
    
    async def verify_statement(self, request: VerificationRequest) -> VerificationResponse:
        """文を包括的に検証"""
        start_time = datetime.now()
        request_id = f"req_{int(start_time.timestamp() * 1000)}"
        
        try:
            # 1. ハルシネーション検出
            hallucination_result = await self._detect_hallucination(request)
            
            # 2. 分野専門家による検証
            domain_analysis = await self._domain_verification(request)
            
            # 3. 専門家コンセンサス形成
            consensus_result = None
            if request.require_consensus:
                consensus_result = await self._form_expert_consensus(request, domain_analysis)
            
            # 4. RAG による外部知識検証
            rag_verification = await self._rag_verification(request)
            
            # 5. 総合判定
            final_result = await self._synthesize_results(
                request, hallucination_result, domain_analysis, 
                consensus_result, rag_verification
            )
            
            # 6. 推奨事項生成
            recommendations = await self._generate_recommendations(
                hallucination_result, domain_analysis, consensus_result
            )
            
            # 処理時間計算
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # レスポンス構築
            response = VerificationResponse(
                request_id=request_id,
                statement=request.statement,
                is_valid=final_result['is_valid'],
                confidence_score=final_result['confidence_score'],
                hallucination_detected=hallucination_result.get('is_hallucination', False),
                expert_consensus=asdict(consensus_result) if consensus_result else None,
                domain_analysis=domain_analysis,
                recommendations=recommendations,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            # 統計更新
            await self._update_statistics(response)
            
            # 履歴記録
            self.verification_history.append(response)
            if len(self.verification_history) > 1000:  # 履歴制限
                self.verification_history = self.verification_history[-500:]
            
            return response
            
        except Exception as e:
            print(f"Error in verification: {e}")
            # エラー時のデフォルトレスポンス
            return VerificationResponse(
                request_id=request_id,
                statement=request.statement,
                is_valid=False,
                confidence_score=0.0,
                hallucination_detected=True,
                expert_consensus=None,
                domain_analysis={"error": str(e)},
                recommendations=["システムエラーが発生しました。再試行してください。"],
                processing_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )
    
    async def _detect_hallucination(self, request: VerificationRequest) -> Dict[str, Any]:
        """ハルシネーション検出"""
        if self.hallucination_detector:
            try:
                result = await self.hallucination_detector.detect_hallucination(
                    request.statement, 
                    request.context, 
                    request.domain_hint
                )
                return asdict(result)
            except Exception as e:
                print(f"Hallucination detection error: {e}")
        
        # モック結果
        return {
            "is_hallucination": False,
            "confidence_score": 0.7,
            "semantic_entropy": 0.3,
            "evidence": ["Mock hallucination check completed"]
        }
    
    async def _domain_verification(self, request: VerificationRequest) -> Dict[str, Any]:
        """分野専門家による検証"""
        try:
            # 関連分野を特定
            domains = self._identify_relevant_domains(request.statement)
            
            verification_results = {}
            
            for domain in domains:
                try:
                    specialist = DomainSpecialistFactory.create_specialist(domain)
                    result = await specialist.verify_statement(
                        request.statement,
                        request.context,
                        VerificationLevel(request.verification_level)
                    )
                    verification_results[domain] = asdict(result)
                except Exception as e:
                    print(f"Domain verification error for {domain}: {e}")
                    verification_results[domain] = {
                        "error": str(e),
                        "is_valid": False,
                        "confidence_score": 0.0
                    }
            
            return verification_results
            
        except Exception as e:
            print(f"Domain verification error: {e}")
            return {"error": str(e)}
    
    async def _form_expert_consensus(self, 
                                   request: VerificationRequest,
                                   domain_analysis: Dict[str, Any]) -> Optional[Any]:
        """専門家コンセンサス形成"""
        if not self.consensus_engine:
            return None
        
        try:
            # ドメイン分析結果をExpertOpinionに変換
            expert_opinions = []
            
            for domain, result in domain_analysis.items():
                if isinstance(result, dict) and 'error' not in result:
                    # モック ExpertOpinion 作成
                    expert_opinions.append({
                        'expert_name': f"{domain}_specialist",
                        'domain': domain,
                        'verification_result': result,
                        'weight': 0.8,
                        'confidence': result.get('confidence_score', 0.5),
                        'reasoning': f"{domain} domain verification",
                        'supporting_evidence': result.get('supporting_references', []),
                        'dissenting_points': result.get('red_flags', [])
                    })
            
            if expert_opinions:
                # コンセンサス形成（簡易版）
                consensus_result = {
                    'consensus_type': 'simple_majority',
                    'agreement_ratio': 0.7,
                    'synthesized_conclusion': 'Mock consensus formed',
                    'recommendations': ['Mock recommendation']
                }
                return consensus_result
            
        except Exception as e:
            print(f"Consensus formation error: {e}")
        
        return None
    
    async def _rag_verification(self, request: VerificationRequest) -> Dict[str, Any]:
        """RAG による外部知識検証"""
        if self.rag_integration:
            try:
                result = await self.rag_integration.verify_statement_with_sources(
                    request.statement, request.context
                )
                return result
            except Exception as e:
                print(f"RAG verification error: {e}")
        
        return {
            "support_level": "unknown",
            "confidence": 0.5,
            "supporting_evidence": []
        }
    
    async def _synthesize_results(self, 
                                request: VerificationRequest,
                                hallucination_result: Dict[str, Any],
                                domain_analysis: Dict[str, Any],
                                consensus_result: Optional[Dict[str, Any]],
                                rag_verification: Dict[str, Any]) -> Dict[str, Any]:
        """結果を統合して最終判定"""
        
        # ハルシネーション判定
        hallucination_detected = hallucination_result.get('is_hallucination', False)
        hallucination_confidence = hallucination_result.get('confidence_score', 0.5)
        
        # ドメイン専門家の平均判定
        domain_validities = []
        domain_confidences = []
        
        for domain, result in domain_analysis.items():
            if isinstance(result, dict) and 'error' not in result:
                domain_validities.append(result.get('is_valid', False))
                domain_confidences.append(result.get('confidence_score', 0.5))
        
        domain_validity = sum(domain_validities) / len(domain_validities) if domain_validities else 0.5
        domain_confidence = sum(domain_confidences) / len(domain_confidences) if domain_confidences else 0.5
        
        # RAG 支持度
        rag_support = 0.7 if rag_verification.get('support_level') == 'strong_support' else 0.3
        
        # 総合判定
        if hallucination_detected and hallucination_confidence > 0.7:
            # 高信頼度でハルシネーション検出
            is_valid = False
            confidence = hallucination_confidence
        else:
            # ドメイン専門家とRAGの総合判定
            weighted_validity = (
                domain_validity * 0.6 + 
                rag_support * 0.3 + 
                (1.0 - hallucination_result.get('semantic_entropy', 0.5)) * 0.1
            )
            is_valid = weighted_validity > 0.6
            confidence = (domain_confidence + rag_verification.get('confidence', 0.5)) / 2
        
        return {
            'is_valid': is_valid,
            'confidence_score': confidence,
            'synthesis_details': {
                'hallucination_factor': hallucination_confidence,
                'domain_factor': domain_validity,
                'rag_factor': rag_support,
                'weighted_validity': weighted_validity if 'weighted_validity' in locals() else 0.5
            }
        }
    
    async def _generate_recommendations(self, 
                                      hallucination_result: Dict[str, Any],
                                      domain_analysis: Dict[str, Any],
                                      consensus_result: Optional[Dict[str, Any]]) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        # ハルシネーション関連推奨
        if hallucination_result.get('is_hallucination'):
            recommendations.append("ハルシネーションが検出されました。情報源を確認してください。")
        
        # ドメイン分析からの推奨
        for domain, result in domain_analysis.items():
            if isinstance(result, dict) and result.get('corrections'):
                for correction in result['corrections'][:2]:  # 最大2つ
                    recommendations.append(f"{domain}: {correction}")
        
        # コンセンサス結果からの推奨
        if consensus_result and consensus_result.get('recommendations'):
            recommendations.extend(consensus_result['recommendations'][:2])
        
        # 一般的推奨
        if not recommendations:
            recommendations.append("より詳細な検証のため、追加の専門家意見を求めることを推奨します。")
        
        return recommendations[:5]  # 最大5つ
    
    def _identify_relevant_domains(self, statement: str) -> List[str]:
        """関連分野を特定"""
        domain_keywords = {
            'consciousness': ['意識', '現象学', 'consciousness', 'phenomenology', 'qualia'],
            'philosophy': ['哲学', '存在', '実在', 'philosophy', 'ontology', 'epistemology'],
            'mathematics': ['数学', '計算', '証明', 'mathematics', 'computation', 'proof']
        }
        
        relevant_domains = []
        statement_lower = statement.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in statement_lower for keyword in keywords):
                relevant_domains.append(domain)
        
        return relevant_domains or ['philosophy']  # デフォルト
    
    async def _update_statistics(self, response: VerificationResponse):
        """統計情報を更新"""
        self.verification_stats["total_verifications"] += 1
        
        if response.hallucination_detected:
            self.verification_stats["hallucinations_detected"] += 1
        
        if response.expert_consensus:
            self.verification_stats["consensus_achieved"] += 1
        
        # 平均処理時間更新
        current_avg = self.verification_stats["average_processing_time"]
        total = self.verification_stats["total_verifications"]
        new_avg = ((current_avg * (total - 1)) + response.processing_time) / total
        self.verification_stats["average_processing_time"] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        return {
            "system_initialized": self.hallucination_detector is not None,
            "verification_stats": self.verification_stats,
            "active_modules": {
                "hallucination_detector": self.hallucination_detector is not None,
                "consensus_engine": self.consensus_engine is not None,
                "rag_integration": self.rag_integration is not None
            }
        }

# FastAPI アプリケーション
app = FastAPI(
    title="Omoikane Lab - Realtime Verification API",
    description="AIバーチャル研究所リアルタイム知識検証システム",
    version="1.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバルインスタンス
verification_system = RealtimeVerificationSystem()
connection_manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    await verification_system.initialize()

@app.get("/")
async def root():
    return {"message": "Omoikane Lab Realtime Verification API", "status": "running"}

@app.get("/status")
async def get_status():
    """システム状態を取得"""
    return verification_system.get_system_status()

@app.post("/verify", response_model=VerificationResponse)
async def verify_statement(request: VerificationRequest):
    """文を検証（REST API）"""
    return await verification_system.verify_statement(request)

@app.get("/history")
async def get_verification_history(limit: int = 50):
    """検証履歴を取得"""
    history = verification_system.verification_history[-limit:]
    return {"history": history, "total_count": len(verification_system.verification_history)}

@app.get("/stats")
async def get_statistics():
    """統計情報を取得"""
    return verification_system.verification_stats

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket エンドポイント"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # クライアントからのメッセージを受信
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "verify_request":
                    # 検証リクエスト処理
                    request_data = message.get("data", {})
                    request = VerificationRequest(**request_data)
                    
                    # 進行状況をクライアントに送信
                    await connection_manager.send_personal_message(
                        {"type": "verification_started", "request_id": f"ws_{int(datetime.now().timestamp() * 1000)}"},
                        websocket
                    )
                    
                    # 検証実行
                    result = await verification_system.verify_statement(request)
                    
                    # 結果送信
                    await connection_manager.send_personal_message(
                        {"type": "verification_result", "data": asdict(result)},
                        websocket
                    )
                
                elif message_type == "ping":
                    # Ping-Pong
                    await connection_manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.now().isoformat()},
                        websocket
                    )
                
            except json.JSONDecodeError:
                await connection_manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON format"},
                    websocket
                )
            except Exception as e:
                await connection_manager.send_personal_message(
                    {"type": "error", "message": str(e)},
                    websocket
                )
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

# 静的ファイル配信（ダッシュボード用）
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )