"""
アプリケーション層

Clean Architecture原則に従ったアプリケーション層の実装。
ユースケース、サービス、DTOを統合し、視覚記号認識システムの
高次機能を提供する。

構成要素:
- Use Cases: ビジネスワークフローの制御
- Services: ドメインロジックの統合
- DTOs: データ転送オブジェクト

谷口忠大の記号創発理論に基づく機能:
- 適応的視覚認識
- 記号創発制御
- メタ認知的記憶管理
"""

from .use_cases import (
    RecognizeImageUseCase,
    TrainVisualSymbolsUseCase, 
    QueryVisualMemoryUseCase
)

from .services import (
    VisualFeatureExtractionService,
    SymbolEmergenceOrchestrationService
)

from .dtos import (
    ImageRecognitionRequest,
    ImageRecognitionResponse,
    SymbolLearningRequest,
    SymbolLearningResponse,
    VisualMemoryQueryRequest,
    VisualMemoryQueryResponse,
    QueryType,
    SortOrder
)

__all__ = [
    # Use Cases
    'RecognizeImageUseCase',
    'TrainVisualSymbolsUseCase', 
    'QueryVisualMemoryUseCase',
    
    # Services
    'VisualFeatureExtractionService',
    'SymbolEmergenceOrchestrationService',
    
    # DTOs
    'ImageRecognitionRequest',
    'ImageRecognitionResponse',
    'SymbolLearningRequest',
    'SymbolLearningResponse',
    'VisualMemoryQueryRequest',
    'VisualMemoryQueryResponse',
    'QueryType',
    'SortOrder'
]