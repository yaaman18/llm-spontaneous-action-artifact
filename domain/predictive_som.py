"""
予測符号化統合SOM実装
JAXベースの予測符号化コアとの理論的統合

エナクティビズム × 予測処理の統合実装
"""
import jax
import jax.numpy as jnp
from jax import vmap, jit, grad
from typing import Dict, List, Tuple, Any, Optional

from .enactive_som import EnactiveSOM, EnactivePattern, ConceptualAffordance


class PredictiveEnactiveSOM(EnactiveSOM):
    """
    予測符号化統合エナクティブSOM
    
    統合要素：
    1. 予測符号化による階層的予測
    2. エナクティブな行為-知覚ループ  
    3. 自由エネルギー原理による更新
    4. 現象学的時間構造
    """
    
    def __init__(self,
                 map_dimensions: Tuple[int, int], 
                 input_dim: int,
                 embodiment_constraints: Dict[str, float],
                 temporal_horizon: int = 10,
                 learning_rate: float = 0.01,
                 prediction_depth: int = 3):
        """
        Args:
            prediction_depth: 予測階層の深さ
            learning_rate: 学習率
        """
        super().__init__(map_dimensions, input_dim, embodiment_constraints, temporal_horizon)
        
        self.learning_rate = learning_rate
        self.prediction_depth = prediction_depth
        
        # 予測階層の初期化
        self.key, *subkeys = jax.random.split(self.key, prediction_depth + 1)
        self.prediction_layers = []
        
        for i, subkey in enumerate(subkeys):
            layer_dim = max(input_dim // (2**i), 4)  # 階層的に次元削減
            layer_weights = jax.random.normal(
                subkey, 
                (map_dimensions[0], map_dimensions[1], layer_dim)
            )
            self.prediction_layers.append(layer_weights)
            
        # 予測誤差の履歴
        self.prediction_error_history = []
        
        # 現在の自由エネルギー
        self._current_free_energy = 0.0
        
    def enactive_update(self,
                       sensory_input: jnp.ndarray,
                       motor_command: jnp.ndarray, 
                       prediction_error: jnp.ndarray) -> 'PredictiveEnactiveSOM':
        """
        エナクティブ更新：感覚-運動統合 + 予測符号化
        """
        # 1. 予測計算
        predictions = self._hierarchical_prediction(sensory_input)
        
        # 2. 予測誤差の計算
        errors = []
        current_error = sensory_input
        
        for i, (prediction, layer_weights) in enumerate(zip(predictions, self.prediction_layers)):
            layer_error = current_error - prediction
            errors.append(layer_error)
            
            # 次の階層のための誤差伝播
            if i < len(predictions) - 1:
                current_error = self._downsample(layer_error)
                
        # 3. 運動指令による予測調整
        motor_adjusted_predictions = self._motor_prediction_adjustment(
            predictions, motor_command
        )
        
        # 4. エナクティブ学習による重み更新
        updated_som = self._update_weights_enactive(
            sensory_input, motor_command, errors, motor_adjusted_predictions
        )
        
        # 5. アフォーダンスマップの更新
        updated_som = updated_som._update_affordances(sensory_input, motor_command)
        
        # 6. 予測誤差履歴の更新
        total_error = sum(jnp.mean(jnp.abs(error)) for error in errors)
        updated_som.prediction_error_history.append(float(total_error))
        
        # 履歴サイズの制限
        if len(updated_som.prediction_error_history) > self.temporal_horizon * 2:
            updated_som.prediction_error_history = updated_som.prediction_error_history[-self.temporal_horizon:]
            
        return updated_som
    
    @jit
    def _hierarchical_prediction(self, input_data: jnp.ndarray) -> List[jnp.ndarray]:
        """階層的予測の計算"""
        predictions = []
        current_input = input_data
        
        for layer_weights in self.prediction_layers:
            # 各SOMニューロンとの距離計算
            distances = vmap(vmap(lambda w: jnp.linalg.norm(current_input - w)))(layer_weights)
            
            # ガウシアン活性化
            sigma = 1.0
            activations = jnp.exp(-distances**2 / (2 * sigma**2))
            
            # 活性化による重み付き予測
            prediction = jnp.sum(
                layer_weights * activations[:, :, jnp.newaxis], 
                axis=(0, 1)
            ) / jnp.sum(activations)
            
            predictions.append(prediction)
            current_input = prediction  # 次の階層への入力
            
        return predictions
    
    @jit
    def _motor_prediction_adjustment(self,
                                   predictions: List[jnp.ndarray],
                                   motor_command: jnp.ndarray) -> List[jnp.ndarray]:
        """運動指令による予測の調整"""
        adjusted_predictions = []
        
        for i, prediction in enumerate(predictions):
            # 運動指令の影響度（階層が深いほど小さい）
            motor_influence = 0.5 / (i + 1)
            
            # 運動指令による予測調整
            motor_adjustment = motor_command * motor_influence
            
            # 次元合わせ
            if len(motor_adjustment) != len(prediction):
                if len(motor_adjustment) > len(prediction):
                    motor_adjustment = motor_adjustment[:len(prediction)]
                else:
                    motor_adjustment = jnp.pad(
                        motor_adjustment, 
                        (0, len(prediction) - len(motor_adjustment))
                    )
                    
            adjusted_prediction = prediction + motor_adjustment
            adjusted_predictions.append(adjusted_prediction)
            
        return adjusted_predictions
    
    def _update_weights_enactive(self,
                               sensory_input: jnp.ndarray,
                               motor_command: jnp.ndarray,
                               errors: List[jnp.ndarray],
                               adjusted_predictions: List[jnp.ndarray]) -> 'PredictiveEnactiveSOM':
        """エナクティブ学習による重み更新"""
        # 新しいSOMインスタンスを作成
        updated_som = PredictiveEnactiveSOM(
            self.map_dimensions,
            self.input_dim,
            self.embodiment_constraints,
            self.temporal_horizon,
            self.learning_rate,
            self.prediction_depth
        )
        
        # 現在の状態をコピー
        updated_som.weight_map = self.weight_map
        updated_som.prediction_layers = self.prediction_layers.copy()
        updated_som.affordance_map = self.affordance_map
        updated_som.prediction_map = self.prediction_map
        updated_som.embodiment_map = self.embodiment_map
        
        # 各階層の重み更新
        for i, (error, adjusted_pred) in enumerate(zip(errors, adjusted_predictions)):
            layer_weights = updated_som.prediction_layers[i]
            
            # 勝者ニューロンの計算
            distances = vmap(vmap(
                lambda w: jnp.linalg.norm(adjusted_pred - w)
            ))(layer_weights)
            winner_idx = jnp.unravel_index(jnp.argmin(distances), distances.shape)
            
            # 近傍関数（ガウシアン）
            h, w = jnp.mgrid[0:self.map_dimensions[0], 0:self.map_dimensions[1]]
            neighborhood = jnp.exp(-((h - winner_idx[0])**2 + (w - winner_idx[1])**2) / (2 * 2.0**2))
            
            # 重み更新
            weight_update = self.learning_rate * neighborhood[:, :, jnp.newaxis] * error[jnp.newaxis, jnp.newaxis, :]
            updated_som.prediction_layers[i] = layer_weights + weight_update
            
        # メイン重みマップの更新（最下位層から）
        bottom_layer_update = (
            self.learning_rate * 
            (sensory_input[jnp.newaxis, jnp.newaxis, :] - updated_som.weight_map)
        )
        updated_som.weight_map += bottom_layer_update
        
        return updated_som
    
    def _update_affordances(self, 
                          sensory_input: jnp.ndarray,
                          motor_command: jnp.ndarray) -> 'PredictiveEnactiveSOM':
        """アフォーダンスマップの更新"""
        # 感覚-運動統合によるアフォーダンスの計算
        sensory_motor_correlation = jnp.outer(sensory_input, motor_command)
        
        # 4方向への行為可能性を推定
        affordance_updates = jnp.zeros_like(self.affordance_map)
        
        for direction in range(4):
            # 各方向への運動コストを計算
            direction_vector = jnp.array([
                jnp.cos(direction * jnp.pi / 2),
                jnp.sin(direction * jnp.pi / 2)
            ])
            
            # 運動指令との内積で方向適合性を計算
            if len(motor_command) >= 2:
                direction_affinity = jnp.dot(motor_command[:2], direction_vector)
                affordance_updates = affordance_updates.at[:, :, direction].set(direction_affinity)
        
        # アフォーダンスマップの更新
        updated_affordance_map = (
            0.9 * self.affordance_map + 
            0.1 * affordance_updates
        )
        
        # 新しいインスタンスを作成して返す
        updated_som = self._create_updated_copy()
        updated_som.affordance_map = updated_affordance_map
        return updated_som
    
    def compute_affordances(self,
                          current_state: jnp.ndarray,
                          context: Dict[str, Any]) -> List[ConceptualAffordance]:
        """概念的アフォーダンスの計算"""
        affordances = []
        
        # 現在状態に最も近いSOMニューロンを特定
        distances = vmap(vmap(
            lambda w: jnp.linalg.norm(current_state - w)
        ))(self.weight_map)
        
        # 上位Kニューロンを選択
        k = 5
        flat_distances = distances.flatten()
        top_k_indices = jnp.argsort(flat_distances)[:k]
        
        for idx in top_k_indices:
            h, w = jnp.unravel_index(idx, distances.shape)
            
            # 概念ベクトル
            concept_vector = self.weight_map[h, w, :]
            
            # 行為ポテンシャル（アフォーダンスマップから）
            action_potential = self.affordance_map[h, w, :]
            
            # 文脈的関連性
            context_relevance = context.get('relevance_weight', 1.0)
            
            # 時間的地平
            temporal_horizon = context.get('temporal_scope', self.temporal_horizon)
            
            affordance = ConceptualAffordance(
                concept_vector=concept_vector,
                action_potential=action_potential,
                contextual_relevance=context_relevance,
                temporal_horizon=temporal_horizon
            )
            affordances.append(affordance)
            
        return affordances
    
    def _predict_next_observation(self,
                                current_obs: jnp.ndarray,
                                action: jnp.ndarray) -> jnp.ndarray:
        """次の観測を予測"""
        # 予測符号化階層を通じた予測
        predictions = self._hierarchical_prediction(current_obs)
        base_prediction = predictions[0] if predictions else current_obs
        
        # 行動による修正
        action_effect = action * 0.1  # 行動の影響度
        
        # 次元合わせ
        if len(action_effect) != len(base_prediction):
            if len(action_effect) > len(base_prediction):
                action_effect = action_effect[:len(base_prediction)]
            else:
                action_effect = jnp.pad(
                    action_effect,
                    (0, len(base_prediction) - len(action_effect))
                )
                
        next_prediction = base_prediction + action_effect
        return next_prediction
    
    def _compute_surprise(self, observation: jnp.ndarray) -> float:
        """観測のサプライズ（負の対数尤度）を計算"""
        # SOMによる観測の予測確率
        distances = vmap(vmap(
            lambda w: jnp.linalg.norm(observation - w)
        ))(self.weight_map)
        
        # ガウシアン確率密度
        sigma = 1.0
        probabilities = jnp.exp(-distances**2 / (2 * sigma**2))
        total_probability = jnp.sum(probabilities)
        
        # サプライズ = -log(確率)
        surprise = -jnp.log(total_probability + 1e-8)
        return float(surprise)
    
    def _compute_complexity(self, action: jnp.ndarray) -> float:
        """行動の複雑性（KLダイバージェンス）を計算"""
        # 行動の事前分布（0平均ガウシアン）からのKLダイバージェンス
        prior_variance = 1.0
        action_variance = jnp.var(action) + 1e-8
        
        kl_divergence = 0.5 * (
            jnp.log(prior_variance / action_variance) + 
            action_variance / prior_variance +
            jnp.sum(action**2) / prior_variance - len(action)
        )
        
        return float(jnp.maximum(kl_divergence, 0.0))
    
    @jit
    def _downsample(self, data: jnp.ndarray) -> jnp.ndarray:
        """データのダウンサンプリング"""
        # 簡単な平均プーリング
        if len(data) > 4:
            reshaped = data[:len(data)//2*2].reshape(-1, 2)
            return jnp.mean(reshaped, axis=1)
        return data
    
    def _create_updated_copy(self) -> 'PredictiveEnactiveSOM':
        """更新されたコピーを作成"""
        new_som = PredictiveEnactiveSOM(
            self.map_dimensions,
            self.input_dim, 
            self.embodiment_constraints,
            self.temporal_horizon,
            self.learning_rate,
            self.prediction_depth
        )
        
        # 状態をコピー
        new_som.weight_map = self.weight_map
        new_som.prediction_layers = self.prediction_layers.copy()
        new_som.affordance_map = self.affordance_map
        new_som.prediction_map = self.prediction_map
        new_som.embodiment_map = self.embodiment_map
        new_som.prediction_error_history = self.prediction_error_history.copy()
        new_som._current_free_energy = self._current_free_energy
        
        return new_som