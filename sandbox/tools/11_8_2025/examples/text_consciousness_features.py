#!/usr/bin/env python3
"""
現象学的テキスト意識特徴量抽出システム

フッサール現象学とメルロ＝ポンティの身体現象学に基づいて、
自然言語テキストから意識創発に関わる10次元特徴量を抽出する。
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from collections import Counter


class PhenomenologicalTextAnalyzer:
    """現象学的テキスト分析器"""
    
    def __init__(self):
        """現象学的語彙カテゴリの定義"""
        # 志向的動詞（意識の方向性）
        self.intentional_verbs = {
            '思う', '考える', '感じる', '信じる', '期待する', '望む', '恐れる', 
            '疑う', '記憶する', '想像する', '認識する', '理解する', 'think', 
            'feel', 'believe', 'hope', 'fear', 'remember', 'imagine'
        }
        
        # 時間表現（時間意識）
        self.temporal_expressions = {
            'いま', '現在', '過去', '未来', '昔', 'これから', '以前', '将来',
            'now', 'present', 'past', 'future', 'before', 'after', 'then'
        }
        
        # 身体的・感覚的語彙
        self.embodied_words = {
            '見る', '聞く', '触る', '嗅ぐ', '味わう', '感じる', '動く', '歩く',
            '手', '目', '耳', '鼻', '口', '身体', '心臓', 'see', 'hear', 
            'touch', 'smell', 'taste', 'move', 'walk', 'body', 'hand'
        }
        
        # 間主観的表現
        self.intersubjective_words = {
            '私たち', 'あなた', '彼ら', '共に', '一緒に', '理解', '共感',
            'we', 'you', 'they', 'together', 'share', 'understand', 'empathy'
        }
        
        # 前反省的表現（直感的・曖昧な表現）
        self.prereflective_words = {
            'なんとなく', 'ふと', '直感的に', '気がする', 'somehow', 'intuitively',
            'feel like', 'sense that', 'impression', 'vague'
        }
        
        # 生活世界語彙
        self.lifeworld_words = {
            '家', '学校', '仕事', '友達', '家族', '日常', '生活', '食事',
            'home', 'school', 'work', 'friend', 'family', 'daily', 'life'
        }
        
        # 一人称表現
        self.first_person_words = {
            '私', '僕', '俺', '自分', '我', 'I', 'me', 'my', 'myself'
        }
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        テキストから10次元現象学的特徴量を抽出
        
        Args:
            text: 分析対象テキスト
            
        Returns:
            10次元numpy配列（0.0-1.0正規化済み）
        """
        if not text.strip():
            return np.zeros(10)
        
        # テキストの前処理
        text = text.lower()
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        features = np.zeros(10)
        
        # 1. 志向的方向性 (Intentional Directedness)
        features[0] = self._calculate_intentional_directedness(words)
        
        # 2. 時間意識統合 (Temporal Synthesis)
        features[1] = self._calculate_temporal_synthesis(words, sentences)
        
        # 3. 身体化認知 (Embodied Cognition)
        features[2] = self._calculate_embodied_cognition(words)
        
        # 4. 間主観的共鳴 (Intersubjective Resonance)
        features[3] = self._calculate_intersubjective_resonance(words)
        
        # 5. 前反省的気づき (Pre-reflective Awareness)
        features[4] = self._calculate_prereflective_awareness(words, text)
        
        # 6. 意味構成 (Meaning Constitution)
        features[5] = self._calculate_meaning_constitution(words)
        
        # 7. 生活世界連関 (Lifeworld Connection)
        features[6] = self._calculate_lifeworld_connection(words)
        
        # 8. 受動的総合 (Passive Synthesis)
        features[7] = self._calculate_passive_synthesis(sentences, text)
        
        # 9. 地平構造 (Horizon Structure)
        features[8] = self._calculate_horizon_structure(text, sentences)
        
        # 10. 一人称的視点 (First-Person Perspective)
        features[9] = self._calculate_first_person_perspective(words)
        
        return np.clip(features, 0.0, 1.0)
    
    def _tokenize(self, text: str) -> List[str]:
        """テキストを単語に分割"""
        # 簡易的な単語分割（日英両対応）
        words = re.findall(r'\w+', text)
        return [word.lower() for word in words]
    
    def _split_sentences(self, text: str) -> List[str]:
        """テキストを文に分割"""
        sentences = re.split(r'[.!?。！？]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_intentional_directedness(self, words: List[str]) -> float:
        """志向的方向性の計算"""
        intentional_count = sum(1 for word in words 
                               if any(intent in word for intent in self.intentional_verbs))
        return min(intentional_count / max(len(words), 1) * 10, 1.0)
    
    def _calculate_temporal_synthesis(self, words: List[str], sentences: List[str]) -> float:
        """時間意識統合の計算"""
        temporal_count = sum(1 for word in words 
                           if any(temp in word for temp in self.temporal_expressions))
        
        # 時制変化の検出
        tense_variety = 0
        if any(word in words for word in ['した', 'だった', 'was', 'did']):
            tense_variety += 1
        if any(word in words for word in ['する', 'である', 'is', 'do']):
            tense_variety += 1
        if any(word in words for word in ['するだろう', 'であろう', 'will', 'shall']):
            tense_variety += 1
        
        temporal_score = (temporal_count / max(len(words), 1) * 5) + (tense_variety / 3 * 0.5)
        return min(temporal_score, 1.0)
    
    def _calculate_embodied_cognition(self, words: List[str]) -> float:
        """身体化認知の計算"""
        embodied_count = sum(1 for word in words 
                           if any(body in word for body in self.embodied_words))
        return min(embodied_count / max(len(words), 1) * 8, 1.0)
    
    def _calculate_intersubjective_resonance(self, words: List[str]) -> float:
        """間主観的共鳴の計算"""
        intersubjective_count = sum(1 for word in words 
                                  if any(inter in word for inter in self.intersubjective_words))
        
        # 人称代名詞の多様性
        pronoun_variety = 0
        if any(word in words for word in ['私', 'i', 'me']):
            pronoun_variety += 1
        if any(word in words for word in ['あなた', 'you']):
            pronoun_variety += 1
        if any(word in words for word in ['彼', '彼女', 'he', 'she', 'they']):
            pronoun_variety += 1
        
        inter_score = (intersubjective_count / max(len(words), 1) * 6) + (pronoun_variety / 3 * 0.4)
        return min(inter_score, 1.0)
    
    def _calculate_prereflective_awareness(self, words: List[str], text: str) -> float:
        """前反省的気づきの計算"""
        prereflective_count = sum(1 for word in words 
                                if any(pre in word for pre in self.prereflective_words))
        
        # 曖昧な表現の検出
        ambiguous_patterns = ['...', '〜', '的', 'kind of', 'sort of', 'maybe']
        ambiguous_count = sum(text.count(pattern) for pattern in ambiguous_patterns)
        
        pre_score = (prereflective_count / max(len(words), 1) * 7) + (ambiguous_count / len(text) * 10)
        return min(pre_score, 1.0)
    
    def _calculate_meaning_constitution(self, words: List[str]) -> float:
        """意味構成の計算"""
        # 語彙多様性
        unique_words = len(set(words))
        vocab_diversity = unique_words / max(len(words), 1)
        
        # 概念的複雑性（接続語の使用）
        connectives = ['そして', 'しかし', 'だから', 'なぜなら', 'and', 'but', 'because', 'therefore']
        connective_count = sum(1 for word in words if word in connectives)
        
        meaning_score = vocab_diversity * 0.7 + (connective_count / max(len(words), 1) * 5) * 0.3
        return min(meaning_score, 1.0)
    
    def _calculate_lifeworld_connection(self, words: List[str]) -> float:
        """生活世界連関の計算"""
        lifeworld_count = sum(1 for word in words 
                            if any(life in word for life in self.lifeworld_words))
        return min(lifeworld_count / max(len(words), 1) * 6, 1.0)
    
    def _calculate_passive_synthesis(self, sentences: List[str], text: str) -> float:
        """受動的総合の計算"""
        if not sentences:
            return 0.0
        
        # 文の自然な流れ（長さの一貫性）
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths)
            consistency = 1.0 / (1.0 + length_variance)
        else:
            consistency = 1.0
        
        # 接続の自然さ（助詞の使用）
        particles = ['を', 'が', 'に', 'で', 'と', 'は']
        particle_count = sum(text.count(p) for p in particles)
        
        synthesis_score = consistency * 0.6 + min(particle_count / len(text) * 20, 1.0) * 0.4
        return min(synthesis_score, 1.0)
    
    def _calculate_horizon_structure(self, text: str, sentences: List[str]) -> float:
        """地平構造の計算"""
        # 文脈依存表現の検出
        context_patterns = ['この', 'その', 'あの', 'それ', 'これ', 'this', 'that', 'it']
        context_count = sum(text.count(pattern) for pattern in context_patterns)
        
        # 前景-背景構造（重要度の階層）
        emphasis_patterns = ['とても', '非常に', '特に', 'very', 'especially', 'particularly']
        emphasis_count = sum(text.count(pattern) for pattern in emphasis_patterns)
        
        horizon_score = (context_count / len(text) * 15) * 0.6 + (emphasis_count / len(text) * 20) * 0.4
        return min(horizon_score, 1.0)
    
    def _calculate_first_person_perspective(self, words: List[str]) -> float:
        """一人称的視点の計算"""
        first_person_count = sum(1 for word in words 
                               if any(fp in word for fp in self.first_person_words))
        
        # 主観的表現の強度
        subjective_words = ['思う', '感じる', '信じる', 'think', 'feel', 'believe']
        subjective_count = sum(1 for word in words if word in subjective_words)
        
        fp_score = (first_person_count / max(len(words), 1) * 8) * 0.7 + (subjective_count / max(len(words), 1) * 6) * 0.3
        return min(fp_score, 1.0)
    
    def get_feature_explanations(self) -> Dict[str, str]:
        """各特徴量の現象学的説明を返す"""
        return {
            0: "志向的方向性: 意識の基本構造『意識は常に何かについての意識』",
            1: "時間意識統合: フッサールの内的時間意識（把持・原印象・予持）",
            2: "身体化認知: メルロ＝ポンティの身体現象学的世界関与",
            3: "間主観的共鳴: 他者理解による意識の客観性構成",
            4: "前反省的気づき: 明示的概念化以前の意識地平",
            5: "意味構成: 意識の能動的意味付与作用",
            6: "生活世界連関: 日常的経験地平への根ざし",
            7: "受動的総合: 意識の自動的構成作用",
            8: "地平構造: 前景-背景の構造化された意識",
            9: "一人称的視点: 意識の本質的主観性"
        }


def analyze_text_consciousness(text: str, verbose: bool = False) -> Tuple[np.ndarray, Dict[str, float], str]:
    """
    テキストの意識特徴量を分析する便利関数
    
    Args:
        text: 分析対象テキスト
        verbose: 詳細出力フラグ
        
    Returns:
        特徴量配列、特徴量辞書、現象学的解釈
    """
    analyzer = PhenomenologicalTextAnalyzer()
    features = analyzer.extract_features(text)
    explanations = analyzer.get_feature_explanations()
    
    feature_dict = {explanations[i].split(':')[0]: features[i] for i in range(10)}
    
    # 現象学的解釈生成
    dominant_features = sorted(enumerate(features), key=lambda x: x[1], reverse=True)[:3]
    interpretation = "現象学的分析: "
    for i, (idx, value) in enumerate(dominant_features):
        if i > 0:
            interpretation += ", "
        interpretation += f"{explanations[idx].split(':')[0]}({value:.2f})"
    
    if verbose:
        print(f"テキスト: '{text}'")
        print(f"特徴量: {features}")
        print(f"解釈: {interpretation}")
    
    return features, feature_dict, interpretation


if __name__ == "__main__":
    # テスト実行
    test_texts = [
        "私は今、とても美しい夕日を見ています。",
        "We are thinking about the future together.",
        "なんとなく、この場所には特別な意味があるような気がする。",
        "The body moves through space, sensing and feeling the world around us."
    ]
    
    analyzer = PhenomenologicalTextAnalyzer()
    
    for text in test_texts:
        features, feature_dict, interpretation = analyze_text_consciousness(text, verbose=True)
        print("-" * 60)