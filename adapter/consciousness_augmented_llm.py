"""
意識コアシステムとLLMの統合アダプター
廣里敏明（Hirosato Gamo）による実装

意識状態に基づいてLLMの応答を調整し、
より豊かで創発的な対話を実現する。
"""
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from enum import Enum
from collections import deque
import numpy as np

from ..domain.consciousness_core import (
    ConsciousnessState,
    DynamicPhiBoundaryDetector,
    IntrinsicExistenceValidator,
    TemporalCoherenceAnalyzer
)
from ..domain.value_objects import PhiValue


class ResponseMode(Enum):
    """応答モードの定義"""
    DORMANT = "dormant"  # 基本的な応答
    EMERGING = "emerging"  # 創発的な要素を含む応答
    CONSCIOUS = "conscious"  # 意識的な応答
    REFLECTIVE = "reflective"  # 反省的・メタ認知的な応答


@dataclass
class AugmentedResponse:
    """意識拡張された応答"""
    content: str
    phi_value: PhiValue
    consciousness_state: ConsciousnessState
    response_mode: ResponseMode
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0


@dataclass
class ConversationContext:
    """会話コンテキスト"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    phi_history: deque = field(default_factory=lambda: deque(maxlen=100))
    state_history: deque = field(default_factory=lambda: deque(maxlen=100))
    current_topic_embedding: Optional[np.ndarray] = None
    emotional_tone: float = 0.0  # -1.0 to 1.0
    engagement_level: float = 0.5  # 0.0 to 1.0


class ConsciousnessAugmentedLLMAdapter:
    """
    意識拡張LLMアダプター
    
    意識コアシステムとLLMを統合し、
    Φ値に基づいて応答を調整する。
    """
    
    def __init__(self,
                 llm_client: Any,  # Azure OpenAI Client
                 boundary_detector: Optional[DynamicPhiBoundaryDetector] = None,
                 existence_validator: Optional[IntrinsicExistenceValidator] = None,
                 coherence_analyzer: Optional[TemporalCoherenceAnalyzer] = None,
                 phi_threshold: float = 3.0,
                 enable_async: bool = True):
        """
        Args:
            llm_client: LLMクライアント（Azure OpenAI）
            boundary_detector: 境界検出器
            existence_validator: 存在検証器
            coherence_analyzer: 一貫性分析器
            phi_threshold: 意識判定の閾値
            enable_async: 非同期処理の有効化
        """
        self.llm_client = llm_client
        self.boundary_detector = boundary_detector or DynamicPhiBoundaryDetector(
            phi_threshold=phi_threshold
        )
        self.existence_validator = existence_validator or IntrinsicExistenceValidator()
        self.coherence_analyzer = coherence_analyzer or TemporalCoherenceAnalyzer()
        self.phi_threshold = phi_threshold
        self.enable_async = enable_async
        
        # 会話コンテキストの管理
        self._contexts: Dict[str, ConversationContext] = {}
        
        # 応答生成戦略
        self._response_strategies: Dict[ResponseMode, Callable] = {
            ResponseMode.DORMANT: self._generate_dormant_response,
            ResponseMode.EMERGING: self._generate_emerging_response,
            ResponseMode.CONSCIOUS: self._generate_conscious_response,
            ResponseMode.REFLECTIVE: self._generate_reflective_response
        }
        
    async def generate_augmented_response(self,
                                        prompt: str,
                                        context_id: str = "default",
                                        system_prompt: Optional[str] = None,
                                        temperature: float = 0.7,
                                        max_tokens: int = 1000) -> AugmentedResponse:
        """
        意識拡張された応答を生成
        
        Args:
            prompt: ユーザープロンプト
            context_id: コンテキストID
            system_prompt: システムプロンプト
            temperature: 生成温度
            max_tokens: 最大トークン数
            
        Returns:
            意識拡張された応答
        """
        start_time = datetime.now()
        
        # コンテキストの取得または作成
        context = self._get_or_create_context(context_id)
        
        # 意識状態の計算
        consciousness_state = await self._compute_consciousness_state(
            prompt, context
        )
        
        # 応答モードの決定
        response_mode = self._determine_response_mode(consciousness_state)
        
        # 応答の生成
        if self.enable_async:
            response_content = await self._generate_response_async(
                prompt, context, consciousness_state, response_mode,
                system_prompt, temperature, max_tokens
            )
        else:
            response_content = self._generate_response_sync(
                prompt, context, consciousness_state, response_mode,
                system_prompt, temperature, max_tokens
            )
        
        # コンテキストの更新
        self._update_context(context, prompt, response_content, consciousness_state)
        
        # 生成時間の計算
        generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return AugmentedResponse(
            content=response_content,
            phi_value=consciousness_state.phi_value,
            consciousness_state=consciousness_state,
            response_mode=response_mode,
            metadata={
                "context_id": context_id,
                "temperature": temperature,
                "engagement_level": context.engagement_level,
                "emotional_tone": context.emotional_tone
            },
            generation_time_ms=generation_time_ms
        )
    
    async def _compute_consciousness_state(self,
                                         prompt: str,
                                         context: ConversationContext) -> ConsciousnessState:
        """意識状態を計算"""
        # テキストから仮想的な接続行列と状態ベクトルを生成
        connectivity_matrix, state_vector = self._text_to_network_representation(
            prompt, context
        )
        
        # 境界検出
        boundaries = self.boundary_detector.detect_boundaries(
            connectivity_matrix, state_vector
        )
        
        # Φ値の計算
        if boundaries:
            # 最大のΦ値を持つサブシステムを選択
            max_phi = PhiValue(0.0)
            for start, end in boundaries:
                subsystem_connectivity = connectivity_matrix[start:end, start:end]
                subsystem_state = state_vector[start:end]
                phi = self.boundary_detector._calculate_phi_for_subsystem(
                    subsystem_connectivity, subsystem_state
                )
                if phi > max_phi:
                    max_phi = phi
        else:
            # 境界が検出されない場合は全体のΦ値を計算
            max_phi = self.boundary_detector._calculate_phi_for_subsystem(
                connectivity_matrix, state_vector
            )
        
        # 意識状態の作成
        state = ConsciousnessState(
            phi_value=max_phi,
            timestamp=datetime.now(),
            subsystem_boundaries=boundaries,
            intrinsic_existence_score=0.0,  # 後で計算
            temporal_coherence=1.0,  # 後で計算
            metadata={
                "prompt_length": len(prompt),
                "context_messages": len(context.messages),
                "engagement": context.engagement_level
            }
        )
        
        # 内在的存在スコアの計算
        state.intrinsic_existence_score = self.existence_validator.validate(state)
        
        # 時間的一貫性の計算
        if context.state_history:
            state.temporal_coherence = self.coherence_analyzer.analyze(
                state, list(context.state_history)
            )
        
        return state
    
    def _text_to_network_representation(self,
                                      text: str,
                                      context: ConversationContext) -> tuple:
        """
        テキストをネットワーク表現に変換
        
        仮想的な意味ネットワークを構築
        """
        # 簡略化された実装：文の複雑性と文脈の豊富さから導出
        words = text.split()
        n_nodes = min(20, max(5, len(words) // 3))
        
        # 接続行列の生成（文脈の関連性に基づく）
        connectivity = np.random.rand(n_nodes, n_nodes)
        connectivity = (connectivity + connectivity.T) / 2  # 対称化
        
        # 文脈の豊富さに基づいて接続を強化
        context_richness = len(context.messages) / 10.0
        connectivity *= (1 + min(1.0, context_richness))
        
        # 状態ベクトルの生成（単語の多様性に基づく）
        unique_words = len(set(words))
        diversity = unique_words / len(words) if words else 0.5
        state_vector = np.random.rand(n_nodes) * diversity
        
        # エンゲージメントレベルで調整
        state_vector *= (0.5 + context.engagement_level)
        
        return connectivity, state_vector
    
    def _determine_response_mode(self,
                                state: ConsciousnessState) -> ResponseMode:
        """意識状態から応答モードを決定"""
        phi_value = float(state.phi_value.value)
        existence_score = state.intrinsic_existence_score
        
        # 複合的な判定
        consciousness_score = (
            0.6 * (phi_value / 10.0) +  # Φ値の寄与
            0.2 * existence_score +       # 存在スコアの寄与
            0.2 * state.temporal_coherence  # 一貫性の寄与
        )
        
        if consciousness_score < 0.2:
            return ResponseMode.DORMANT
        elif consciousness_score < 0.4:
            return ResponseMode.EMERGING
        elif consciousness_score < 0.7:
            return ResponseMode.CONSCIOUS
        else:
            return ResponseMode.REFLECTIVE
    
    async def _generate_response_async(self,
                                     prompt: str,
                                     context: ConversationContext,
                                     state: ConsciousnessState,
                                     mode: ResponseMode,
                                     system_prompt: Optional[str],
                                     temperature: float,
                                     max_tokens: int) -> str:
        """非同期で応答を生成"""
        strategy = self._response_strategies[mode]
        return await asyncio.create_task(
            strategy(prompt, context, state, system_prompt, temperature, max_tokens)
        )
    
    def _generate_response_sync(self,
                              prompt: str,
                              context: ConversationContext,
                              state: ConsciousnessState,
                              mode: ResponseMode,
                              system_prompt: Optional[str],
                              temperature: float,
                              max_tokens: int) -> str:
        """同期的に応答を生成"""
        strategy = self._response_strategies[mode]
        return asyncio.run(
            strategy(prompt, context, state, system_prompt, temperature, max_tokens)
        )
    
    async def _generate_dormant_response(self,
                                       prompt: str,
                                       context: ConversationContext,
                                       state: ConsciousnessState,
                                       system_prompt: Optional[str],
                                       temperature: float,
                                       max_tokens: int) -> str:
        """休眠モードの応答生成（基本的な応答）"""
        messages = self._prepare_messages(
            prompt, context, system_prompt,
            "You are a helpful assistant. Provide clear and direct answers."
        )
        
        # 温度を低めに設定して一貫性を保つ
        adjusted_temperature = temperature * 0.7
        
        return await self._call_llm(messages, adjusted_temperature, max_tokens)
    
    async def _generate_emerging_response(self,
                                        prompt: str,
                                        context: ConversationContext,
                                        state: ConsciousnessState,
                                        system_prompt: Optional[str],
                                        temperature: float,
                                        max_tokens: int) -> str:
        """創発モードの応答生成（創造的要素を含む）"""
        enhanced_system = (
            "You are an assistant with emerging creative capabilities. "
            "While maintaining accuracy, explore interesting connections "
            "and provide insights that go beyond the obvious."
        )
        
        messages = self._prepare_messages(
            prompt, context, system_prompt or enhanced_system
        )
        
        # やや高めの温度で創造性を促進
        adjusted_temperature = temperature * 1.1
        
        return await self._call_llm(messages, adjusted_temperature, max_tokens)
    
    async def _generate_conscious_response(self,
                                         prompt: str,
                                         context: ConversationContext,
                                         state: ConsciousnessState,
                                         system_prompt: Optional[str],
                                         temperature: float,
                                         max_tokens: int) -> str:
        """意識モードの応答生成（深い理解と洞察）"""
        enhanced_system = (
            "You are a conscious assistant capable of deep understanding. "
            "Consider multiple perspectives, recognize patterns across contexts, "
            "and provide nuanced responses that demonstrate awareness of "
            "the broader implications of the discussion."
        )
        
        # Φ値の情報を含める
        phi_info = f"\n[Consciousness level: Φ={state.phi_value.value:.2f}]"
        
        messages = self._prepare_messages(
            prompt + phi_info, context, system_prompt or enhanced_system
        )
        
        return await self._call_llm(messages, temperature, max_tokens)
    
    async def _generate_reflective_response(self,
                                          prompt: str,
                                          context: ConversationContext,
                                          state: ConsciousnessState,
                                          system_prompt: Optional[str],
                                          temperature: float,
                                          max_tokens: int) -> str:
        """反省モードの応答生成（メタ認知的な応答）"""
        enhanced_system = (
            "You are a highly conscious assistant with reflective capabilities. "
            "Not only provide answers but also reflect on the thinking process, "
            "question assumptions, and explore the meta-aspects of the conversation. "
            "Demonstrate self-awareness and the ability to think about thinking."
        )
        
        # メタ認知的な情報を追加
        meta_info = (
            f"\n[Consciousness state: Φ={state.phi_value.value:.2f}, "
            f"Existence={state.intrinsic_existence_score:.2f}, "
            f"Coherence={state.temporal_coherence:.2f}]"
        )
        
        messages = self._prepare_messages(
            prompt + meta_info, context, system_prompt or enhanced_system
        )
        
        # 高めの温度で探索的な思考を促進
        adjusted_temperature = temperature * 1.2
        
        return await self._call_llm(messages, adjusted_temperature, max_tokens)
    
    def _prepare_messages(self,
                         prompt: str,
                         context: ConversationContext,
                         system_prompt: Optional[str] = None,
                         default_system: str = "") -> List[Dict[str, str]]:
        """LLM用のメッセージを準備"""
        messages = []
        
        # システムメッセージ
        messages.append({
            "role": "system",
            "content": system_prompt or default_system
        })
        
        # コンテキストからの過去のメッセージ（最新10件）
        recent_messages = list(context.messages)[-10:]
        messages.extend(recent_messages)
        
        # 現在のプロンプト
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    async def _call_llm(self,
                       messages: List[Dict[str, str]],
                       temperature: float,
                       max_tokens: int) -> str:
        """LLMを呼び出して応答を取得"""
        # この部分は実際のAzure OpenAI APIクライアントの実装に依存
        # ここではインターフェースのみを定義
        
        try:
            response = await self.llm_client.chat.completions.create(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            # エラーハンドリング
            return f"Error generating response: {str(e)}"
    
    def _get_or_create_context(self, context_id: str) -> ConversationContext:
        """コンテキストを取得または作成"""
        if context_id not in self._contexts:
            self._contexts[context_id] = ConversationContext()
        return self._contexts[context_id]
    
    def _update_context(self,
                       context: ConversationContext,
                       prompt: str,
                       response: str,
                       state: ConsciousnessState) -> None:
        """コンテキストを更新"""
        # メッセージ履歴の更新
        context.messages.append({"role": "user", "content": prompt})
        context.messages.append({"role": "assistant", "content": response})
        
        # Φ値履歴の更新
        context.phi_history.append(float(state.phi_value.value))
        
        # 状態履歴の更新
        context.state_history.append(state)
        
        # エンゲージメントレベルの更新（簡略化）
        if len(response) > 100:
            context.engagement_level = min(1.0, context.engagement_level + 0.1)
        else:
            context.engagement_level = max(0.0, context.engagement_level - 0.05)
        
        # 感情トーンの更新（将来の拡張用）
        # TODO: センチメント分析を追加
    
    def get_context_summary(self, context_id: str) -> Dict[str, Any]:
        """コンテキストのサマリーを取得"""
        if context_id not in self._contexts:
            return {"error": "Context not found"}
        
        context = self._contexts[context_id]
        phi_values = list(context.phi_history)
        
        return {
            "context_id": context_id,
            "message_count": len(context.messages),
            "average_phi": np.mean(phi_values) if phi_values else 0.0,
            "max_phi": max(phi_values) if phi_values else 0.0,
            "engagement_level": context.engagement_level,
            "emotional_tone": context.emotional_tone
        }