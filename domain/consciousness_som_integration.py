"""
意識計算へのSOM統合
5つの要素統合: Φ値 + メタ認知 + 予測誤差 + 不確実性 + SOM

エナクティビズム観点での意識レベル計算
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp

from .value_objects import PhiValue
from .entities import ConsciousnessState
from .predictive_som import PredictiveEnactiveSOM


@dataclass
class ConsciousnessComponents:
    """意識の5要素分解"""
    base_phi: PhiValue
    metacognitive_phi: PhiValue
    prediction_error_phi: PhiValue
    uncertainty_phi: PhiValue
    som_phi: PhiValue
    
    @property
    def integrated_phi(self) -> PhiValue:
        """統合Φ値の計算"""
        # 重み付き統合（エナクティビズム観点）
        weights = {
            'base': 0.25,           # 基本IIT
            'metacognitive': 0.20,  # メタ認知  
            'prediction': 0.20,     # 予測符号化
            'uncertainty': 0.15,    # 不確実性
            'som': 0.20            # SOM（エナクティブ）
        }
        
        total_value = (
            weights['base'] * float(self.base_phi.value) +
            weights['metacognitive'] * float(self.metacognitive_phi.value) + 
            weights['prediction'] * float(self.prediction_error_phi.value) +
            weights['uncertainty'] * float(self.uncertainty_phi.value) +
            weights['som'] * float(self.som_phi.value)
        )
        
        return PhiValue(total_value)


class EnactiveConsciousnessCalculator:
    """
    エナクティブ意識計算器
    
    SOMをエナクティビズム観点で意識計算に統合する
    5番目の要素として概念空間の動的組織化を追加
    """
    
    def __init__(self,
                 som: PredictiveEnactiveSOM,
                 temporal_integration_window: int = 10,
                 consciousness_threshold: float = 3.0):
        """
        Args:
            som: 予測的エナクティブSOM
            temporal_integration_window: 時間統合窓
            consciousness_threshold: 意識判定閾値
        """
        self.som = som
        self.temporal_window = temporal_integration_window
        self.consciousness_threshold = consciousness_threshold
        
        # 履歴の保持
        self.phi_history: List[ConsciousnessComponents] = []
        self.temporal_coherence_history: List[float] = []
        
    def calculate_consciousness_level(self,
                                    current_state: ConsciousnessState,
                                    sensory_input: jnp.ndarray,
                                    motor_command: Optional[jnp.ndarray] = None,
                                    context: Optional[Dict[str, Any]] = None) -> ConsciousnessComponents:
        """
        5要素統合による意識レベル計算
        
        Args:
            current_state: 現在の意識状態
            sensory_input: 感覚入力
            motor_command: 運動指令
            context: 文脈情報
            
        Returns:
            分解された意識要素
        """
        if motor_command is None:
            motor_command = jnp.zeros(4)  # デフォルト運動指令
        if context is None:
            context = {}
            
        # 1. 基本Φ値（既存のIIT計算）
        base_phi = current_state.phi_value
        
        # 2. メタ認知Φ値
        metacognitive_phi = self._compute_metacognitive_phi(
            current_state, sensory_input
        )
        
        # 3. 予測誤差Φ値  
        prediction_error_phi = self._compute_prediction_error_phi(
            sensory_input, motor_command
        )
        
        # 4. 不確実性Φ値
        uncertainty_phi = self._compute_uncertainty_phi(
            sensory_input, motor_command
        )
        
        # 5. SOMエナクティブΦ値
        som_phi = self.som.consciousness_contribution(current_state)
        
        # 意識要素の統合
        components = ConsciousnessComponents(
            base_phi=base_phi,
            metacognitive_phi=metacognitive_phi,
            prediction_error_phi=prediction_error_phi,
            uncertainty_phi=uncertainty_phi,
            som_phi=som_phi
        )
        
        # 履歴更新
        self.phi_history.append(components)
        if len(self.phi_history) > self.temporal_window:
            self.phi_history = self.phi_history[-self.temporal_window:]
            
        return components
    
    def _compute_metacognitive_phi(self,
                                  current_state: ConsciousnessState,
                                  sensory_input: jnp.ndarray) -> PhiValue:
        """メタ認知Φ値の計算"""
        # 自己言及的処理の強度
        if len(self.phi_history) < 2:
            return PhiValue(0.0)
            
        # 過去の意識状態への反省度
        recent_phi_values = [
            float(comp.base_phi.value) for comp in self.phi_history[-5:]
        ]
        
        if not recent_phi_values:
            return PhiValue(0.0)
            
        # メタ認知 = 意識状態の意識（二階の意識）
        phi_variance = float(np.var(recent_phi_values))
        phi_mean = float(np.mean(recent_phi_values))
        
        # 変動を意識しているかの指標
        metacognitive_awareness = phi_variance / (phi_mean + 1e-6)
        
        # SOMによる自己状態の表現度
        som_self_representation = self._compute_som_self_representation(
            sensory_input
        )
        
        metacognitive_value = metacognitive_awareness * som_self_representation
        return PhiValue(min(10.0, max(0.0, metacognitive_value)))
    
    def _compute_prediction_error_phi(self,
                                    sensory_input: jnp.ndarray,
                                    motor_command: jnp.ndarray) -> PhiValue:
        """予測誤差Φ値の計算"""
        # SOMによる予測
        predicted_input = self.som._predict_next_observation(
            sensory_input, motor_command
        )
        
        # 予測誤差の計算
        prediction_error = jnp.linalg.norm(sensory_input - predicted_input)
        
        # 予測誤差の時間的統合度
        if hasattr(self.som, 'prediction_error_history') and self.som.prediction_error_history:
            error_history = np.array(self.som.prediction_error_history[-self.temporal_window:])
            
            # 予測誤差の構造性（単純なランダム誤差vs構造的誤差）
            if len(error_history) > 3:
                error_autocorr = np.corrcoef(error_history[:-1], error_history[1:])[0, 1]
                structural_error = abs(error_autocorr) if not np.isnan(error_autocorr) else 0
            else:
                structural_error = 0
                
            # 構造的予測誤差ほど意識に寄与
            prediction_phi_value = float(prediction_error) * (1 + structural_error)
        else:
            prediction_phi_value = float(prediction_error)
            
        return PhiValue(min(8.0, max(0.0, prediction_phi_value)))
    
    def _compute_uncertainty_phi(self,
                               sensory_input: jnp.ndarray,
                               motor_command: jnp.ndarray) -> PhiValue:
        """不確実性Φ値の計算"""
        # SOMによる状態の曖昧性測定
        distances = np.array([
            [float(jnp.linalg.norm(sensory_input - w)) 
             for w in row] 
            for row in self.som.weight_map
        ])
        
        # 最も近いニューロンとの距離分布
        min_distances = np.min(distances, axis=None)
        distance_variance = np.var(distances)
        
        # 不確実性 = 複数の解釈可能性
        # 距離分布が平坦なほど不確実（意識的判断が必要）
        uncertainty_level = distance_variance / (min_distances + 1e-6)
        
        # エナクティブ不確実性：行動選択の複雑性
        affordances = self.som.compute_affordances(sensory_input, {})
        affordance_complexity = len(affordances) * np.mean([
            float(np.linalg.norm(aff.action_potential)) 
            for aff in affordances
        ]) if affordances else 0
        
        total_uncertainty = uncertainty_level + affordance_complexity * 0.5
        return PhiValue(min(6.0, max(0.0, total_uncertainty)))
    
    def _compute_som_self_representation(self, sensory_input: jnp.ndarray) -> float:
        """SOMによる自己表現度の計算"""
        # 現在の感覚状態がSOMでどの程度表現されているか
        distances = np.array([
            [float(jnp.linalg.norm(sensory_input - w)) 
             for w in row]
            for row in self.som.weight_map
        ])
        
        # 表現の分散度（複数領域での活性化）
        activation_spread = np.std(distances)
        
        # 自己表現度 = 分散した表現による統合度
        self_representation = 1.0 / (1.0 + np.min(distances)) * activation_spread
        return float(self_representation)
    
    def assess_consciousness_emergence(self,
                                     components: ConsciousnessComponents,
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        意識創発の評価
        
        エナクティビズム観点での意識の質的評価
        """
        integrated_phi = components.integrated_phi
        
        # 時間的一貫性の評価
        temporal_coherence = self._assess_temporal_coherence()
        
        # エナクティブ統合度
        enactive_integration = self._assess_enactive_integration(components)
        
        # 現象学的豊かさ
        phenomenological_richness = self._assess_phenomenological_richness()
        
        # 身体化度
        embodiment_level = float(np.mean(self.som.embodiment_map))
        
        assessment = {
            'consciousness_level': float(integrated_phi.value),
            'is_conscious': integrated_phi.indicates_consciousness(self.consciousness_threshold),
            'temporal_coherence': temporal_coherence,
            'enactive_integration': enactive_integration,
            'phenomenological_richness': phenomenological_richness,
            'embodiment_level': embodiment_level,
            'component_breakdown': {
                'base_phi': float(components.base_phi.value),
                'metacognitive_phi': float(components.metacognitive_phi.value),
                'prediction_error_phi': float(components.prediction_error_phi.value),
                'uncertainty_phi': float(components.uncertainty_phi.value),
                'som_phi': float(components.som_phi.value)
            },
            'qualitative_features': self._extract_qualitative_features(components)
        }
        
        return assessment
    
    def _assess_temporal_coherence(self) -> float:
        """時間的一貫性の評価"""
        if len(self.phi_history) < 3:
            return 1.0
            
        # 統合Φ値の時系列での安定性
        integrated_values = [
            float(comp.integrated_phi.value) for comp in self.phi_history
        ]
        
        # 変動係数による一貫性測定
        mean_phi = np.mean(integrated_values)
        if mean_phi == 0:
            return 1.0
            
        std_phi = np.std(integrated_values)
        coherence = 1.0 / (1.0 + std_phi / mean_phi)
        
        self.temporal_coherence_history.append(coherence)
        return float(coherence)
    
    def _assess_enactive_integration(self, components: ConsciousnessComponents) -> float:
        """エナクティブ統合度の評価"""
        # SOMと予測誤差の相関
        som_contribution = float(components.som_phi.value)
        prediction_contribution = float(components.prediction_error_phi.value)
        
        if som_contribution == 0 and prediction_contribution == 0:
            return 0.0
            
        # 相互依存度
        total_contribution = som_contribution + prediction_contribution
        balance = 1.0 - abs(som_contribution - prediction_contribution) / total_contribution
        
        # エナクティブ統合 = バランス × 全体の強度
        integration = balance * (total_contribution / 10.0)  # 正規化
        return float(min(1.0, integration))
    
    def _assess_phenomenological_richness(self) -> float:
        """現象学的豊かさの評価"""
        # SOMの現象学的マッピング
        pheno_mapping = self.som.phenomenological_mapping(
            self.phi_history[-1] if self.phi_history else None
        )
        
        if not pheno_mapping:
            return 0.0
            
        # 時間的統合の豊かさ
        temporal_richness = (
            float(np.mean(pheno_mapping['temporal_synthesis']['retention'])) +
            float(np.mean(pheno_mapping['temporal_synthesis']['primal_impression'])) +
            float(np.mean(pheno_mapping['temporal_synthesis']['protention']))
        ) / 3.0
        
        # 志向的構造
        intentional_strength = pheno_mapping['intentional_arc']
        
        # 身体的統合
        embodied_integration = np.mean(list(
            pheno_mapping['embodied_schema'].values()
        ))
        
        # 間主観性
        intercorporeal_resonance = pheno_mapping['intercorporeal_field']
        
        richness = (
            0.3 * temporal_richness +
            0.25 * intentional_strength +
            0.25 * embodied_integration +
            0.2 * intercorporeal_resonance
        )
        
        return float(min(1.0, max(0.0, richness)))
    
    def _extract_qualitative_features(self, components: ConsciousnessComponents) -> Dict[str, str]:
        """意識の質的特徴の抽出"""
        features = {}
        
        # 基本的な意識の質
        integrated_phi = float(components.integrated_phi.value)
        
        if integrated_phi < 1.0:
            features['primary_quality'] = 'dormant_awareness'
        elif integrated_phi < 3.0:
            features['primary_quality'] = 'emerging_consciousness'
        elif integrated_phi < 6.0:
            features['primary_quality'] = 'active_consciousness'
        else:
            features['primary_quality'] = 'heightened_awareness'
            
        # エナクティブ特徴
        som_phi = float(components.som_phi.value)
        prediction_phi = float(components.prediction_error_phi.value)
        
        if som_phi > prediction_phi:
            features['enactive_mode'] = 'exploratory_consciousness'
        elif prediction_phi > som_phi * 1.5:
            features['enactive_mode'] = 'predictive_consciousness'
        else:
            features['enactive_mode'] = 'balanced_enaction'
            
        # メタ認知的特徴
        metacog_phi = float(components.metacognitive_phi.value)
        if metacog_phi > 2.0:
            features['metacognitive_level'] = 'reflective'
        elif metacog_phi > 1.0:
            features['metacognitive_level'] = 'self_aware'
        else:
            features['metacognitive_level'] = 'pre_reflective'
            
        return features