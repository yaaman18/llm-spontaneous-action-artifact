#!/usr/bin/env python3
"""
NewbornAI 2.0 リアルタイム検証APIサーバー
ハルシネーション検出と包括的ドキュメント検証のためのRESTful API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import yaml
import uuid
from datetime import datetime, timedelta
import logging
from pathlib import Path
import aiofiles
import hashlib

# 内部モジュール
from consensus_engine import ConsensusEngine, ConsensusResult, ValidationResult

app = FastAPI(
    title="NewbornAI 2.0 Verification API",
    description="包括的ドキュメント検証・ハルシネーション検出システム",
    version="2.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル変数
consensus_engine = ConsensusEngine()
verification_cache = {}
active_verifications = {}

# データモデル
class DocumentVerificationRequest(BaseModel):
    document_content: str = Field(..., description="検証対象ドキュメントの内容")
    document_name: str = Field(..., description="ドキュメント名")
    verification_level: str = Field(default="comprehensive", description="検証レベル (basic/standard/comprehensive)")
    focus_areas: List[str] = Field(default=[], description="重点検証領域")
    expert_domains: List[str] = Field(default=[], description="対象専門領域")

class BatchVerificationRequest(BaseModel):
    document_paths: List[str] = Field(..., description="検証対象ドキュメントパスリスト")
    verification_config: Dict[str, Any] = Field(default={}, description="検証設定")

class VerificationResult(BaseModel):
    verification_id: str
    document_name: str
    overall_confidence: float
    consensus_strength: float
    identified_hallucinations: List[Dict[str, Any]]
    theoretical_issues: List[Dict[str, Any]]
    implementation_concerns: List[Dict[str, Any]]
    recommendations: List[str]
    expert_assessments: List[Dict[str, Any]]
    final_assessment: Dict[str, float]
    processing_time: float
    timestamp: str

class VerificationStatus(BaseModel):
    verification_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0-1.0
    current_stage: str
    estimated_completion: Optional[str]
    error_message: Optional[str]

# APIエンドポイント

@app.get("/", summary="API概要取得")
async def root():
    """API基本情報"""
    return {
        "service": "NewbornAI 2.0 Verification API",
        "version": "2.0.0",
        "description": "包括的ドキュメント検証・ハルシネーション検出システム",
        "features": [
            "multi_expert_consensus",
            "hallucination_detection", 
            "theoretical_accuracy_validation",
            "implementation_feasibility_assessment",
            "real_time_verification",
            "batch_processing"
        ],
        "endpoints": {
            "single_verification": "/verify/document",
            "batch_verification": "/verify/batch",
            "file_upload": "/verify/upload",
            "status_check": "/status/{verification_id}",
            "results": "/results/{verification_id}"
        }
    }

@app.post("/verify/document", response_model=VerificationResult, summary="単一ドキュメント検証")
async def verify_document(
    request: DocumentVerificationRequest,
    background_tasks: BackgroundTasks
):
    """
    単一ドキュメントのリアルタイム検証
    """
    verification_id = str(uuid.uuid4())
    
    try:
        # 検証開始
        start_time = datetime.now()
        active_verifications[verification_id] = {
            "status": "processing",
            "progress": 0.0,
            "start_time": start_time,
            "current_stage": "initialization"
        }
        
        logger.info(f"Starting verification {verification_id} for document: {request.document_name}")
        
        # 一時ファイル作成
        temp_file_path = f"/tmp/verification_{verification_id}.md"
        async with aiofiles.open(temp_file_path, 'w', encoding='utf-8') as f:
            await f.write(request.document_content)
        
        # 専門家コンセンサス形成
        active_verifications[verification_id]["current_stage"] = "expert_consensus"
        active_verifications[verification_id]["progress"] = 0.3
        
        consensus_results = await consensus_engine.form_consensus([temp_file_path])
        
        if temp_file_path not in consensus_results:
            raise HTTPException(status_code=500, detail="Verification processing failed")
        
        result = consensus_results[temp_file_path]
        
        # 結果変換
        active_verifications[verification_id]["current_stage"] = "result_processing"
        active_verifications[verification_id]["progress"] = 0.8
        
        expert_assessments = []
        for opinion in result.expert_opinions:
            expert_assessments.append({
                "expert_name": opinion.expert_name,
                "domain": opinion.domain.value,
                "confidence_score": opinion.confidence_score,
                "theoretical_accuracy": opinion.theoretical_accuracy,
                "implementation_feasibility": opinion.implementation_feasibility,
                "consistency_score": opinion.consistency_score,
                "hallucination_risk": opinion.hallucination_risk,
                "identified_issues_count": len(opinion.identified_issues)
            })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        verification_result = VerificationResult(
            verification_id=verification_id,
            document_name=request.document_name,
            overall_confidence=result.overall_confidence,
            consensus_strength=result.consensus_strength,
            identified_hallucinations=result.identified_hallucinations,
            theoretical_issues=result.theoretical_issues,
            implementation_concerns=result.implementation_concerns,
            recommendations=result.recommendations,
            expert_assessments=expert_assessments,
            final_assessment=result.final_assessment,
            processing_time=processing_time,
            timestamp=result.timestamp.isoformat()
        )
        
        # 完了状態更新
        active_verifications[verification_id]["status"] = "completed"
        active_verifications[verification_id]["progress"] = 1.0
        active_verifications[verification_id]["result"] = verification_result
        
        # キャッシュ保存
        verification_cache[verification_id] = verification_result
        
        # 一時ファイル削除
        Path(temp_file_path).unlink(missing_ok=True)
        
        logger.info(f"Verification {verification_id} completed in {processing_time:.2f}s")
        
        return verification_result
        
    except Exception as e:
        logger.error(f"Verification {verification_id} failed: {str(e)}")
        active_verifications[verification_id]["status"] = "failed"
        active_verifications[verification_id]["error_message"] = str(e)
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.post("/verify/batch", summary="バッチドキュメント検証")
async def verify_batch(
    request: BatchVerificationRequest,
    background_tasks: BackgroundTasks
):
    """
    複数ドキュメントのバッチ検証
    """
    verification_id = str(uuid.uuid4())
    
    try:
        start_time = datetime.now()
        active_verifications[verification_id] = {
            "status": "processing",
            "progress": 0.0,
            "start_time": start_time,
            "current_stage": "batch_initialization",
            "total_documents": len(request.document_paths)
        }
        
        logger.info(f"Starting batch verification {verification_id} for {len(request.document_paths)} documents")
        
        # バックグラウンドでバッチ処理実行
        background_tasks.add_task(process_batch_verification, verification_id, request)
        
        return {
            "verification_id": verification_id,
            "status": "accepted",
            "message": f"Batch verification started for {len(request.document_paths)} documents",
            "check_status_url": f"/status/{verification_id}",
            "estimated_completion_minutes": len(request.document_paths) * 0.5
        }
        
    except Exception as e:
        logger.error(f"Batch verification {verification_id} failed to start: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch verification failed to start: {str(e)}")

@app.post("/verify/upload", summary="ファイルアップロード検証")
async def verify_upload(
    file: UploadFile = File(...),
    verification_level: str = "comprehensive",
    background_tasks: BackgroundTasks = None
):
    """
    アップロードファイルの検証
    """
    verification_id = str(uuid.uuid4())
    
    try:
        # ファイル内容読み取り
        content = await file.read()
        document_content = content.decode('utf-8')
        
        # 検証リクエスト作成
        verification_request = DocumentVerificationRequest(
            document_content=document_content,
            document_name=file.filename,
            verification_level=verification_level
        )
        
        # 検証実行
        return await verify_document(verification_request, background_tasks)
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload verification failed: {str(e)}")

@app.get("/status/{verification_id}", response_model=VerificationStatus, summary="検証状態確認")
async def get_verification_status(verification_id: str):
    """
    検証プロセスの現在状態を取得
    """
    if verification_id not in active_verifications:
        raise HTTPException(status_code=404, detail="Verification ID not found")
    
    verification_info = active_verifications[verification_id]
    
    # 推定完了時間計算
    estimated_completion = None
    if verification_info["status"] == "processing" and verification_info["progress"] > 0:
        elapsed = (datetime.now() - verification_info["start_time"]).total_seconds()
        estimated_total = elapsed / verification_info["progress"]
        remaining = estimated_total - elapsed
        estimated_completion = (datetime.now() + timedelta(seconds=remaining)).isoformat()
    
    return VerificationStatus(
        verification_id=verification_id,
        status=verification_info["status"],
        progress=verification_info["progress"],
        current_stage=verification_info["current_stage"],
        estimated_completion=estimated_completion,
        error_message=verification_info.get("error_message")
    )

@app.get("/results/{verification_id}", summary="検証結果取得")
async def get_verification_results(verification_id: str):
    """
    完了した検証の結果を取得
    """
    if verification_id in verification_cache:
        return verification_cache[verification_id]
    
    if verification_id in active_verifications:
        verification_info = active_verifications[verification_id]
        if verification_info["status"] == "completed" and "result" in verification_info:
            return verification_info["result"]
        elif verification_info["status"] == "processing":
            raise HTTPException(status_code=202, detail="Verification still in progress")
        elif verification_info["status"] == "failed":
            raise HTTPException(status_code=500, detail=verification_info.get("error_message", "Verification failed"))
    
    raise HTTPException(status_code=404, detail="Verification results not found")

@app.get("/health", summary="ヘルスチェック")
async def health_check():
    """
    APIサーバーのヘルスチェック
    """
    try:
        # 基本的なシステムチェック
        consensus_engine_status = "operational" if consensus_engine else "unavailable"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "consensus_engine": consensus_engine_status,
            "active_verifications": len(active_verifications),
            "cached_results": len(verification_cache),
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/metrics", summary="システムメトリクス")
async def get_metrics():
    """
    システムパフォーマンスメトリクス
    """
    return {
        "active_verifications": len(active_verifications),
        "completed_verifications": len(verification_cache),
        "system_load": {
            "cpu_usage": "N/A",  # 実装時に実際のCPU使用率を取得
            "memory_usage": "N/A",  # 実装時に実際のメモリ使用率を取得
        },
        "average_processing_time": calculate_average_processing_time(),
        "expert_panel_status": {
            "available_experts": len(consensus_engine.experts),
            "expert_domains": [expert.profile.domain.value for expert in consensus_engine.experts]
        }
    }

# バックグラウンド処理関数
async def process_batch_verification(verification_id: str, request: BatchVerificationRequest):
    """
    バッチ検証のバックグラウンド処理
    """
    try:
        start_time = datetime.now()
        total_docs = len(request.document_paths)
        
        active_verifications[verification_id]["current_stage"] = "batch_processing"
        
        # バッチコンセンサス形成
        consensus_results = await consensus_engine.form_consensus(request.document_paths)
        
        # 結果処理
        batch_results = {}
        for i, (doc_path, result) in enumerate(consensus_results.items()):
            progress = (i + 1) / total_docs
            active_verifications[verification_id]["progress"] = progress
            
            batch_results[doc_path] = {
                "overall_confidence": result.overall_confidence,
                "consensus_strength": result.consensus_strength,
                "identified_hallucinations": result.identified_hallucinations,
                "theoretical_issues": result.theoretical_issues,
                "implementation_concerns": result.implementation_concerns,
                "recommendations": result.recommendations,
                "final_assessment": result.final_assessment,
                "expert_count": len(result.expert_opinions)
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 完了状態更新
        active_verifications[verification_id]["status"] = "completed"
        active_verifications[verification_id]["progress"] = 1.0
        active_verifications[verification_id]["result"] = {
            "verification_id": verification_id,
            "batch_results": batch_results,
            "summary": {
                "total_documents": total_docs,
                "processing_time": processing_time,
                "average_confidence": sum(r["overall_confidence"] for r in batch_results.values()) / total_docs,
                "high_confidence_docs": sum(1 for r in batch_results.values() if r["overall_confidence"] > 0.8),
                "total_hallucinations": sum(len(r["identified_hallucinations"]) for r in batch_results.values())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Batch verification {verification_id} completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Batch verification {verification_id} failed: {str(e)}")
        active_verifications[verification_id]["status"] = "failed"
        active_verifications[verification_id]["error_message"] = str(e)

def calculate_average_processing_time() -> float:
    """
    平均処理時間計算
    """
    completed_verifications = [
        v for v in active_verifications.values() 
        if v["status"] == "completed" and "result" in v
    ]
    
    if not completed_verifications:
        return 0.0
    
    processing_times = []
    for verification in completed_verifications:
        if hasattr(verification.get("result"), "processing_time"):
            processing_times.append(verification["result"].processing_time)
    
    return sum(processing_times) / len(processing_times) if processing_times else 0.0

# サーバー起動時の初期化
@app.on_event("startup")
async def startup_event():
    """
    サーバー起動時の初期化処理
    """
    logger.info("NewbornAI 2.0 Verification API Server starting up...")
    logger.info(f"Available experts: {len(consensus_engine.experts)}")
    logger.info("API Server ready for verification requests")

# サーバー終了時のクリーンアップ
@app.on_event("shutdown")
async def shutdown_event():
    """
    サーバー終了時のクリーンアップ処理
    """
    logger.info("NewbornAI 2.0 Verification API Server shutting down...")
    # アクティブな検証の適切な終了処理を実装可能

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )