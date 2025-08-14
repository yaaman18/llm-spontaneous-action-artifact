"""
Multilingual Learning API Endpoints.

RESTful API interface for multilingual learning operations.
Provides HTTP endpoints that expose the application services
following Clean Architecture principles.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import uuid

# Import framework-agnostic API components
# In a real implementation, this would use FastAPI, Flask, or similar
try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Fallback definitions for type hints
    class BaseModel:
        pass
    
    def Field(*args, **kwargs):
        return None

from ...application.services.multilingual_learning_service import MultilingualLearningService
from ...infrastructure.persistence.learning_repository_impl import LearningRepositoryImpl
from ...infrastructure.external.sentencepiece_adapter import SentencePieceAdapter
from ...infrastructure.config.system_config import SystemConfig
from ...domain.value_objects.consciousness_state import ConsciousnessState


# Request/Response Models
class CreateSessionRequest(BaseModel):
    """Request model for creating a learning session."""
    session_name: Optional[str] = Field(None, description="Optional name for the session")
    max_clusters: int = Field(20, ge=1, le=100, description="Maximum number of language clusters")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Cluster similarity threshold")
    boundary_confidence_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Boundary detection confidence threshold")
    learning_rate: float = Field(0.01, ge=0.001, le=1.0, description="Learning rate for adaptation")


class LearnFromTextRequest(BaseModel):
    """Request model for learning from text."""
    text: str = Field(..., min_length=1, max_length=1000000, description="Text to learn from")
    save_progress: bool = Field(True, description="Whether to save learning progress")
    consciousness_integration: bool = Field(False, description="Whether to integrate with consciousness state")


class TokenizeTextRequest(BaseModel):
    """Request model for text tokenization."""
    text: str = Field(..., min_length=1, max_length=10000000, description="Text to tokenize")
    include_metadata: bool = Field(True, description="Include detailed metadata in response")
    use_caching: bool = Field(True, description="Enable result caching")
    consciousness_integration: bool = Field(False, description="Integrate with consciousness state")


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    texts: List[str] = Field(..., min_items=1, max_items=1000, description="List of texts to process")
    batch_size: int = Field(10, ge=1, le=100, description="Number of texts per batch")
    parallel_processing: bool = Field(False, description="Enable parallel processing")
    consciousness_integration: bool = Field(False, description="Integrate with consciousness state")


class SessionResponse(BaseModel):
    """Response model for session operations."""
    session_id: str
    session_name: Optional[str]
    created_at: str
    status: str
    message: str


class LearningResponse(BaseModel):
    """Response model for learning operations."""
    session_id: str
    success: bool
    tokens: List[str]
    processing_time: float
    learning_metrics: Dict[str, Any]
    cluster_id: Optional[str]
    timestamp: str


class TokenizationResponse(BaseModel):
    """Response model for tokenization operations."""
    session_id: str
    success: bool
    tokens: List[str]
    processing_time: float
    metadata: Optional[Dict[str, Any]]
    cache_hit: bool
    timestamp: str


class MultilingualAPI:
    """
    API interface for multilingual learning operations.
    
    This class provides HTTP endpoints for all multilingual learning
    functionality while maintaining clean separation from business logic.
    It acts as a thin adapter layer between HTTP requests and application services.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize API with dependencies.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize infrastructure layer
        self.repository = LearningRepositoryImpl(config)
        self.sentencepiece_adapter = SentencePieceAdapter() if FASTAPI_AVAILABLE else None
        
        # Initialize application layer
        self.learning_service = MultilingualLearningService(self.repository, config)
        
        # Initialize API framework
        if FASTAPI_AVAILABLE:
            self.app = FastAPI(
                title="Multilingual Learning API",
                description="API for multilingual tokenization and learning",
                version="1.0.0"
            )
            self._setup_routes()
        else:
            self.app = None
    
    def _setup_routes(self):
        """Setup API routes."""
        if not self.app:
            return
        
        # Session management endpoints
        self.app.post("/sessions", response_model=SessionResponse)(self.create_session)
        self.app.get("/sessions/{session_id}")(self.get_session_info)
        self.app.get("/sessions")(self.list_sessions)
        self.app.delete("/sessions/{session_id}")(self.delete_session)
        
        # Learning endpoints
        self.app.post("/sessions/{session_id}/learn", response_model=LearningResponse)(self.learn_from_text)
        self.app.post("/sessions/{session_id}/learn/batch")(self.learn_from_batch)
        
        # Tokenization endpoints
        self.app.post("/sessions/{session_id}/tokenize", response_model=TokenizationResponse)(self.tokenize_text)
        self.app.post("/sessions/{session_id}/tokenize/batch")(self.tokenize_batch)
        self.app.post("/sessions/{session_id}/tokenize/analyze")(self.analyze_tokenization)
        
        # Cluster management endpoints
        self.app.get("/sessions/{session_id}/clusters")(self.get_clusters)
        self.app.get("/sessions/{session_id}/clusters/{cluster_id}")(self.get_cluster_details)
        
        # Export/Import endpoints
        self.app.get("/sessions/{session_id}/export")(self.export_session)
        self.app.post("/sessions/import")(self.import_session)
        
        # Checkpoints endpoints
        self.app.post("/sessions/{session_id}/checkpoints/{checkpoint_name}")(self.save_checkpoint)
        self.app.put("/sessions/{session_id}/checkpoints/{checkpoint_name}/load")(self.load_checkpoint)
        
        # Comparison endpoints (SentencePiece integration)
        self.app.post("/sessions/{session_id}/compare/sentencepiece")(self.compare_with_sentencepiece)
        
        # Health and statistics endpoints
        self.app.get("/health")(self.health_check)
        self.app.get("/statistics")(self.get_global_statistics)
        self.app.get("/sessions/{session_id}/statistics")(self.get_session_statistics)
    
    async def create_session(self, request: CreateSessionRequest) -> SessionResponse:
        """Create a new learning session."""
        try:
            tokenizer_config = {
                'max_clusters': request.max_clusters,
                'similarity_threshold': request.similarity_threshold,
                'boundary_confidence_threshold': request.boundary_confidence_threshold,
                'learning_rate': request.learning_rate
            }
            
            session_id = await self.learning_service.create_learning_session(
                session_name=request.session_name,
                tokenizer_config=tokenizer_config
            )
            
            return SessionResponse(
                session_id=session_id,
                session_name=request.session_name,
                created_at=datetime.now().isoformat(),
                status="created",
                message="Session created successfully"
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session."""
        try:
            session_summary = await self.learning_service.get_session_summary(session_id)
            return session_summary
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions."""
        try:
            active_sessions = self.learning_service.get_active_sessions()
            
            return {
                'active_sessions': active_sessions,
                'total_count': len(active_sessions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session."""
        try:
            success = self.learning_service.close_session(session_id)
            
            return {
                'session_id': session_id,
                'deleted': success,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def learn_from_text(
        self,
        session_id: str,
        request: LearnFromTextRequest
    ) -> LearningResponse:
        """Learn from text input."""
        try:
            # Get consciousness state if integration is requested
            consciousness_state = None
            if request.consciousness_integration:
                consciousness_state = self._get_current_consciousness_state()
            
            result = await self.learning_service.learn_from_text(
                session_id=session_id,
                text=request.text,
                consciousness_state=consciousness_state,
                save_progress=request.save_progress
            )
            
            return LearningResponse(
                session_id=session_id,
                success=result.get('success', False),
                tokens=result.get('tokens', []),
                processing_time=result.get('processing_time', 0),
                learning_metrics=result.get('learning_metrics', {}),
                cluster_id=result.get('cluster_id'),
                timestamp=result.get('timestamp', datetime.now().isoformat())
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def learn_from_batch(
        self,
        session_id: str,
        request: BatchProcessRequest
    ) -> Dict[str, Any]:
        """Learn from multiple text samples."""
        try:
            consciousness_state = None
            if request.consciousness_integration:
                consciousness_state = self._get_current_consciousness_state()
            
            results = await self.learning_service.learn_from_batch(
                session_id=session_id,
                text_samples=request.texts,
                consciousness_state=consciousness_state,
                batch_size=request.batch_size
            )
            
            # Calculate summary statistics
            successful_results = [r for r in results if r.get('success', False)]
            total_tokens = sum(len(r.get('tokens', [])) for r in successful_results)
            total_processing_time = sum(r.get('processing_time', 0) for r in results)
            
            return {
                'session_id': session_id,
                'batch_size': len(request.texts),
                'successful_samples': len(successful_results),
                'total_tokens_generated': total_tokens,
                'total_processing_time': total_processing_time,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def tokenize_text(
        self,
        session_id: str,
        request: TokenizeTextRequest
    ) -> TokenizationResponse:
        """Tokenize text."""
        try:
            consciousness_state = None
            if request.consciousness_integration:
                consciousness_state = self._get_current_consciousness_state()
            
            result = await self.learning_service.tokenize_text(
                session_id=session_id,
                text=request.text,
                consciousness_state=consciousness_state,
                include_metadata=request.include_metadata
            )
            
            return TokenizationResponse(
                session_id=session_id,
                success=result.get('success', False),
                tokens=result.get('tokens', []),
                processing_time=result.get('processing_time', 0),
                metadata=result.get('metadata'),
                cache_hit=result.get('cache_hit', False),
                timestamp=result.get('timestamp', datetime.now().isoformat())
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def tokenize_batch(
        self,
        session_id: str,
        request: BatchProcessRequest
    ) -> Dict[str, Any]:
        """Tokenize multiple texts."""
        try:
            consciousness_state = None
            if request.consciousness_integration:
                consciousness_state = self._get_current_consciousness_state()
            
            results = await self.learning_service.tokenize_batch(
                session_id=session_id,
                text_samples=request.texts,
                consciousness_state=consciousness_state,
                parallel_processing=request.parallel_processing
            )
            
            # Calculate summary statistics
            successful_results = [r for r in results if r.get('success', False)]
            total_tokens = sum(len(r.get('tokens', [])) for r in successful_results)
            cache_hits = sum(1 for r in results if r.get('cache_hit', False))
            
            return {
                'session_id': session_id,
                'batch_size': len(request.texts),
                'successful_tokenizations': len(successful_results),
                'total_tokens_generated': total_tokens,
                'cache_hit_rate': cache_hits / len(results) if results else 0,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def analyze_tokenization(
        self,
        session_id: str,
        request: TokenizeTextRequest
    ) -> Dict[str, Any]:
        """Get detailed tokenization analysis."""
        try:
            consciousness_state = None
            if request.consciousness_integration:
                consciousness_state = self._get_current_consciousness_state()
            
            analysis = await self.learning_service.get_tokenization_analysis(
                session_id=session_id,
                text=request.text,
                consciousness_state=consciousness_state
            )
            
            return analysis
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_clusters(self, session_id: str) -> Dict[str, Any]:
        """Get all language clusters for a session."""
        try:
            clusters = await self.learning_service.get_language_clusters(session_id)
            
            return {
                'session_id': session_id,
                'cluster_count': len(clusters),
                'clusters': clusters,
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_cluster_details(
        self,
        session_id: str,
        cluster_id: str
    ) -> Dict[str, Any]:
        """Get details for a specific cluster."""
        try:
            clusters = await self.learning_service.get_language_clusters(session_id)
            
            cluster_details = None
            for cluster in clusters:
                if cluster['cluster_id'] == cluster_id:
                    cluster_details = cluster
                    break
            
            if not cluster_details:
                raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
            
            return {
                'session_id': session_id,
                'cluster_details': cluster_details,
                'timestamp': datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def save_checkpoint(
        self,
        session_id: str,
        checkpoint_name: str
    ) -> Dict[str, Any]:
        """Save a session checkpoint."""
        try:
            success = await self.learning_service.save_session_checkpoint(
                session_id, checkpoint_name
            )
            
            return {
                'session_id': session_id,
                'checkpoint_name': checkpoint_name,
                'saved': success,
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def load_checkpoint(
        self,
        session_id: str,
        checkpoint_name: str
    ) -> Dict[str, Any]:
        """Load a session checkpoint."""
        try:
            success = await self.learning_service.load_learning_session(
                session_id, checkpoint_name
            )
            
            return {
                'session_id': session_id,
                'checkpoint_name': checkpoint_name,
                'loaded': success,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def export_session(
        self,
        session_id: str,
        format: str = Query("json", description="Export format")
    ) -> Dict[str, Any]:
        """Export session data."""
        try:
            exported_data = await self.learning_service.export_session(
                session_id, format
            )
            
            return exported_data
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def import_session(
        self,
        session_data: Dict[str, Any],
        new_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Import session data."""
        try:
            imported_session_id = await self.learning_service.import_session(
                session_data, new_session_id
            )
            
            return {
                'imported_session_id': imported_session_id,
                'original_session_id': session_data.get('session_id'),
                'import_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def compare_with_sentencepiece(
        self,
        session_id: str,
        text: str = Form(...),
        model_name: str = Form(...),
        vocab_size: int = Form(1000)
    ) -> Dict[str, Any]:
        """Compare tokenization with SentencePiece."""
        try:
            if not self.sentencepiece_adapter:
                raise HTTPException(
                    status_code=501,
                    detail="SentencePiece integration not available"
                )
            
            # Get custom tokenization
            tokenization_result = await self.learning_service.tokenize_text(
                session_id=session_id,
                text=text,
                include_metadata=True
            )
            
            custom_tokens = tokenization_result.get('tokens', [])
            
            # Train SentencePiece model if it doesn't exist
            if model_name not in self.sentencepiece_adapter.list_models():
                self.sentencepiece_adapter.train_model_from_text(
                    [text], model_name, vocab_size
                )
            
            # Compare tokenizations
            comparison = self.sentencepiece_adapter.compare_tokenizations(
                text, custom_tokens, model_name
            )
            
            return {
                'session_id': session_id,
                'comparison_analysis': comparison,
                'timestamp': datetime.now().isoformat()
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        try:
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'active_sessions': len(self.learning_service.get_active_sessions()),
                'config': {
                    'framework': self.config.framework.value,
                    'device': self.config.device,
                    'debug_mode': self.config.debug_mode
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Health check failed: {e}")
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """Get global system statistics."""
        try:
            # Get repository statistics
            cluster_stats = await self.repository.get_cluster_statistics()
            
            # Get cache statistics
            cache_stats = self.learning_service.clear_cache()
            
            return {
                'cluster_statistics': cluster_stats,
                'cache_statistics': cache_stats,
                'active_sessions': len(self.learning_service.get_active_sessions()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_session_statistics(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session."""
        try:
            session_summary = await self.learning_service.get_session_summary(session_id)
            return session_summary
            
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_current_consciousness_state(self) -> Optional[ConsciousnessState]:
        """Get current consciousness state for integration."""
        try:
            # In a full implementation, this would retrieve the current
            # consciousness state from the consciousness framework
            # For now, return a minimal consciousness state
            return ConsciousnessState.create_minimal_consciousness()
        except Exception:
            return None
    
    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


# Factory function for creating API instance
def create_api(config: SystemConfig) -> MultilingualAPI:
    """
    Factory function to create API instance.
    
    Args:
        config: System configuration
        
    Returns:
        Configured MultilingualAPI instance
    """
    return MultilingualAPI(config)