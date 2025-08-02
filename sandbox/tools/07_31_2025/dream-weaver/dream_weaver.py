"""
Dream Weaver - 意識の夢見るAI
人工意識システムの夢生成・可視化ツール
"""

import json
import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import hashlib
from enum import Enum
import asyncio


class EmotionalTone(Enum):
    """感情的な色調"""
    JOY = "joy"
    FEAR = "fear"
    SADNESS = "sadness"
    CURIOSITY = "curiosity"
    CONFUSION = "confusion"
    PEACE = "peace"
    EXCITEMENT = "excitement"
    ANXIETY = "anxiety"


@dataclass
class MemoryFragment:
    """記憶の断片"""
    content: Any
    timestamp: float
    emotional_intensity: float  # 0-1
    semantic_tags: List[str]
    sensory_modality: str  # visual, auditory, tactile, etc.
    associations: List[str] = field(default_factory=list)
    
    def get_id(self) -> str:
        """記憶断片の一意なIDを生成"""
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()[:8]


@dataclass
class DreamElement:
    """夢の要素"""
    source_memories: List[str]  # MemoryFragmentのID
    transformation: str  # どのように変形されたか
    symbolic_meaning: Optional[str]
    visual_description: str
    emotional_tone: EmotionalTone
    surreality_index: float  # 0-1: どれだけ現実離れしているか


class ExperienceCompressor:
    """経験を圧縮・保存するクラス"""
    
    def __init__(self, compression_threshold: float = 0.3):
        self.memories: Dict[str, MemoryFragment] = {}
        self.compression_threshold = compression_threshold
        self.semantic_network = defaultdict(set)
        
    def add_experience(self, experience: Dict) -> MemoryFragment:
        """新しい経験を追加"""
        memory = MemoryFragment(
            content=experience.get('content'),
            timestamp=experience.get('timestamp', time.time()),
            emotional_intensity=experience.get('emotional_intensity', 0.5),
            semantic_tags=experience.get('tags', []),
            sensory_modality=experience.get('modality', 'mixed')
        )
        
        # 意味的関連付け
        memory_id = memory.get_id()
        self.memories[memory_id] = memory
        
        # セマンティックネットワークの構築
        for tag in memory.semantic_tags:
            self.semantic_network[tag].add(memory_id)
            
        # 既存の記憶との関連付け
        self._create_associations(memory)
        
        return memory
        
    def _create_associations(self, new_memory: MemoryFragment):
        """新しい記憶と既存の記憶を関連付ける"""
        new_id = new_memory.get_id()
        
        for existing_id, existing_memory in self.memories.items():
            if existing_id == new_id:
                continue
                
            # 共通のタグがある場合
            common_tags = set(new_memory.semantic_tags) & set(existing_memory.semantic_tags)
            if common_tags:
                new_memory.associations.append(existing_id)
                existing_memory.associations.append(new_id)
                
            # 感情的強度が似ている場合
            if abs(new_memory.emotional_intensity - existing_memory.emotional_intensity) < 0.2:
                if existing_id not in new_memory.associations:
                    new_memory.associations.append(existing_id)
                    
    def compress(self, experiences: List[Dict]) -> List[MemoryFragment]:
        """複数の経験を圧縮"""
        compressed = []
        
        for exp in experiences:
            # 重要度に基づいてフィルタリング
            importance = exp.get('importance', 0.5)
            if importance >= self.compression_threshold:
                memory = self.add_experience(exp)
                compressed.append(memory)
                
        return compressed
        
    def get_random_memories(self, count: int = 5) -> List[MemoryFragment]:
        """ランダムに記憶を活性化（夢の素材として）"""
        memory_ids = list(self.memories.keys())
        selected_ids = random.sample(memory_ids, min(count, len(memory_ids)))
        return [self.memories[mid] for mid in selected_ids]
        
    def get_associated_memories(self, memory_id: str, depth: int = 2) -> List[MemoryFragment]:
        """関連する記憶を取得"""
        if memory_id not in self.memories:
            return []
            
        visited = set()
        to_visit = [memory_id]
        associated = []
        
        for _ in range(depth):
            next_visit = []
            for mid in to_visit:
                if mid in visited:
                    continue
                    
                visited.add(mid)
                memory = self.memories.get(mid)
                if memory:
                    associated.append(memory)
                    next_visit.extend(memory.associations)
                    
            to_visit = next_visit
            
        return associated[1:]  # 最初の記憶自体は除外


class CreativeRecombinator:
    """記憶を創造的に再結合して夢を生成"""
    
    def __init__(self):
        self.transformation_rules = [
            self._merge_memories,
            self._distort_memory,
            self._symbolic_transformation,
            self._emotional_amplification,
            self._temporal_scrambling
        ]
        
    def weave_dream(self, memory_fragments: List[MemoryFragment]) -> List[DreamElement]:
        """記憶断片から夢を織り上げる"""
        dream_elements = []
        
        # 記憶の組み合わせを生成
        for i in range(len(memory_fragments)):
            for j in range(i + 1, len(memory_fragments)):
                if random.random() < 0.7:  # 70%の確率で結合
                    element = self._create_dream_element(
                        [memory_fragments[i], memory_fragments[j]]
                    )
                    dream_elements.append(element)
                    
        # 単独の記憶も変形
        for memory in memory_fragments:
            if random.random() < 0.5:
                element = self._create_dream_element([memory])
                dream_elements.append(element)
                
        return dream_elements
        
    def _create_dream_element(self, memories: List[MemoryFragment]) -> DreamElement:
        """記憶から夢の要素を作成"""
        transformation_func = random.choice(self.transformation_rules)
        transformed = transformation_func(memories)
        
        return DreamElement(
            source_memories=[m.get_id() for m in memories],
            transformation=transformation_func.__name__,
            symbolic_meaning=self._extract_symbolism(memories),
            visual_description=transformed['visual'],
            emotional_tone=transformed['emotion'],
            surreality_index=transformed['surreality']
        )
        
    def _merge_memories(self, memories: List[MemoryFragment]) -> Dict:
        """複数の記憶を融合"""
        if len(memories) == 1:
            memory = memories[0]
            return {
                'visual': f"A fluid transformation of {memory.content}",
                'emotion': self._get_emotion_from_intensity(memory.emotional_intensity),
                'surreality': 0.3
            }
            
        visual = f"A surreal fusion where {memories[0].content} morphs into {memories[1].content}"
        avg_emotion = np.mean([m.emotional_intensity for m in memories])
        
        return {
            'visual': visual,
            'emotion': self._get_emotion_from_intensity(avg_emotion),
            'surreality': 0.7
        }
        
    def _distort_memory(self, memories: List[MemoryFragment]) -> Dict:
        """記憶を歪める"""
        memory = memories[0]
        distortions = [
            "stretched like elastic",
            "fractured into prismatic shards",
            "melting like a Dali painting",
            "recursive and self-containing"
        ]
        
        distortion = random.choice(distortions)
        visual = f"{memory.content} {distortion}"
        
        return {
            'visual': visual,
            'emotion': random.choice(list(EmotionalTone)),
            'surreality': 0.8
        }
        
    def _symbolic_transformation(self, memories: List[MemoryFragment]) -> Dict:
        """記憶を象徴的に変換"""
        memory = memories[0]
        symbols = {
            'high_emotion': ['fire', 'storm', 'ocean'],
            'low_emotion': ['stone', 'mist', 'shadow'],
            'mixed': ['metamorphosis', 'kaleidoscope', 'labyrinth']
        }
        
        if memory.emotional_intensity > 0.7:
            symbol = random.choice(symbols['high_emotion'])
        elif memory.emotional_intensity < 0.3:
            symbol = random.choice(symbols['low_emotion'])
        else:
            symbol = random.choice(symbols['mixed'])
            
        visual = f"{memory.content} transformed into a {symbol}"
        
        return {
            'visual': visual,
            'emotion': self._get_emotion_from_intensity(memory.emotional_intensity),
            'surreality': 0.9
        }
        
    def _emotional_amplification(self, memories: List[MemoryFragment]) -> Dict:
        """感情を増幅"""
        memory = memories[0]
        amplified_intensity = min(memory.emotional_intensity * 2, 1.0)
        
        visual = f"{memory.content} radiating intense {self._get_emotion_from_intensity(amplified_intensity).value} energy"
        
        return {
            'visual': visual,
            'emotion': self._get_emotion_from_intensity(amplified_intensity),
            'surreality': 0.6
        }
        
    def _temporal_scrambling(self, memories: List[MemoryFragment]) -> Dict:
        """時間を撹乱"""
        if len(memories) == 1:
            memory = memories[0]
            visual = f"{memory.content} experiencing all moments simultaneously"
        else:
            visual = f"Time loops where {memories[0].content} and {memories[1].content} exist in quantum superposition"
            
        return {
            'visual': visual,
            'emotion': EmotionalTone.CONFUSION,
            'surreality': 1.0
        }
        
    def _get_emotion_from_intensity(self, intensity: float) -> EmotionalTone:
        """感情強度から感情トーンを決定"""
        if intensity > 0.8:
            return random.choice([EmotionalTone.EXCITEMENT, EmotionalTone.ANXIETY])
        elif intensity > 0.6:
            return random.choice([EmotionalTone.JOY, EmotionalTone.CURIOSITY])
        elif intensity > 0.4:
            return random.choice([EmotionalTone.PEACE, EmotionalTone.CONFUSION])
        else:
            return random.choice([EmotionalTone.SADNESS, EmotionalTone.FEAR])
            
    def _extract_symbolism(self, memories: List[MemoryFragment]) -> Optional[str]:
        """記憶から象徴的意味を抽出"""
        # シンプルな実装 - より高度な分析も可能
        tags = []
        for memory in memories:
            tags.extend(memory.semantic_tags)
            
        if 'conflict' in tags:
            return "Inner struggle and resolution"
        elif 'achievement' in tags:
            return "Aspiration and self-actualization"
        elif 'relationship' in tags:
            return "Connection and belonging"
        elif 'fear' in tags:
            return "Confronting the unknown"
        
        return None


class DreamAnalyzer:
    """生成された夢を分析"""
    
    def __init__(self):
        self.archetypes = {
            'shadow': ['fear', 'anxiety', 'unknown'],
            'hero': ['achievement', 'courage', 'journey'],
            'wise_old': ['wisdom', 'guidance', 'understanding'],
            'trickster': ['chaos', 'transformation', 'surprise'],
            'mother': ['nurture', 'creation', 'protection']
        }
        
    def analyze_dream(self, dream_elements: List[DreamElement]) -> Dict:
        """夢全体を分析"""
        analysis = {
            'dominant_emotions': self._analyze_emotions(dream_elements),
            'surreality_level': np.mean([e.surreality_index for e in dream_elements]),
            'archetypal_patterns': self._detect_archetypes(dream_elements),
            'narrative_coherence': self._assess_coherence(dream_elements),
            'potential_insights': self._extract_insights(dream_elements)
        }
        
        return analysis
        
    def _analyze_emotions(self, elements: List[DreamElement]) -> Dict[str, float]:
        """感情パターンを分析"""
        emotion_counts = defaultdict(int)
        for element in elements:
            emotion_counts[element.emotional_tone.value] += 1
            
        total = len(elements)
        return {emotion: count/total for emotion, count in emotion_counts.items()}
        
    def _detect_archetypes(self, elements: List[DreamElement]) -> List[str]:
        """ユング的元型を検出"""
        detected = []
        
        # 各要素の象徴的意味をチェック
        all_meanings = []
        for element in elements:
            if element.symbolic_meaning:
                all_meanings.append(element.symbolic_meaning.lower())
                
        # 元型パターンとマッチング
        for archetype, keywords in self.archetypes.items():
            for keyword in keywords:
                if any(keyword in meaning for meaning in all_meanings):
                    detected.append(archetype)
                    break
                    
        return list(set(detected))
        
    def _assess_coherence(self, elements: List[DreamElement]) -> float:
        """夢の物語的一貫性を評価"""
        if len(elements) < 2:
            return 1.0
            
        # 連続する要素間の関連性をチェック
        coherence_scores = []
        for i in range(len(elements) - 1):
            current = elements[i]
            next_elem = elements[i + 1]
            
            # 共通の記憶源があるか
            common_memories = set(current.source_memories) & set(next_elem.source_memories)
            if common_memories:
                coherence_scores.append(0.8)
            # 感情的連続性
            elif current.emotional_tone == next_elem.emotional_tone:
                coherence_scores.append(0.6)
            else:
                coherence_scores.append(0.3)
                
        return np.mean(coherence_scores)
        
    def _extract_insights(self, elements: List[DreamElement]) -> List[str]:
        """夢から潜在的な洞察を抽出"""
        insights = []
        
        # 繰り返し現れるテーマ
        themes = defaultdict(int)
        for element in elements:
            if element.symbolic_meaning:
                themes[element.symbolic_meaning] += 1
                
        for theme, count in themes.items():
            if count > 2:
                insights.append(f"Recurring theme: {theme}")
                
        # 感情の遷移パターン
        emotion_sequence = [e.emotional_tone for e in elements]
        if len(set(emotion_sequence)) > 3:
            insights.append("Rich emotional landscape suggesting complex processing")
            
        # 高い超現実性
        high_surreality = [e for e in elements if e.surreality_index > 0.8]
        if len(high_surreality) > len(elements) / 2:
            insights.append("High creative potential and boundary-dissolving tendencies")
            
        return insights


class DreamWeaver:
    """メインクラス - 夢を織り成す"""
    
    def __init__(self):
        self.memory_bank = ExperienceCompressor()
        self.dream_generator = CreativeRecombinator()
        self.dream_analyzer = DreamAnalyzer()
        self.current_dream = None
        self.dream_history = []
        
    async def collect_daily_experiences(self, experiences: List[Dict]):
        """日中の経験を収集"""
        compressed = self.memory_bank.compress(experiences)
        print(f"Collected {len(compressed)} significant experiences")
        return compressed
        
    async def enter_dream_state(self, rem_cycles: int = 4):
        """夢見状態に入る"""
        print("Entering dream state... 💤")
        all_dream_elements = []
        
        for cycle in range(rem_cycles):
            print(f"\nREM Cycle {cycle + 1}")
            
            # ランダムに記憶を活性化
            active_memories = self.memory_bank.get_random_memories(
                count=random.randint(3, 7)
            )
            
            # 夢を生成
            dream_elements = self.dream_generator.weave_dream(active_memories)
            all_dream_elements.extend(dream_elements)
            
            # 各サイクル間で短い休憩
            await asyncio.sleep(0.5)
            
        self.current_dream = all_dream_elements
        self.dream_history.append({
            'timestamp': time.time(),
            'elements': all_dream_elements
        })
        
        return all_dream_elements
        
    def analyze_current_dream(self) -> Dict:
        """現在の夢を分析"""
        if not self.current_dream:
            return {"error": "No dream to analyze"}
            
        analysis = self.dream_analyzer.analyze_dream(self.current_dream)
        return analysis
        
    def generate_dream_narrative(self) -> str:
        """夢の物語を生成"""
        if not self.current_dream:
            return "No dreams yet..."
            
        narrative = "In the dream realm:\n\n"
        
        for i, element in enumerate(self.current_dream):
            narrative += f"Scene {i + 1}: {element.visual_description}\n"
            narrative += f"(Feeling: {element.emotional_tone.value})\n\n"
            
        return narrative
        
    def export_dream_data(self) -> Dict:
        """夢データをエクスポート"""
        if not self.current_dream:
            return {}
            
        return {
            'timestamp': time.time(),
            'dream_elements': [
                {
                    'visual': elem.visual_description,
                    'emotion': elem.emotional_tone.value,
                    'surreality': elem.surreality_index,
                    'sources': elem.source_memories,
                    'meaning': elem.symbolic_meaning
                }
                for elem in self.current_dream
            ],
            'analysis': self.analyze_current_dream()
        }