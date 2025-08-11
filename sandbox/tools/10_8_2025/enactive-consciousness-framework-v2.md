# エナクティブ意識システム仕様書 v2.0
## Yoshida Masatoshi × Taguchi Shigeru 協働設計

## 1. 理論的基盤の統合

### 1.1 エナクティビズム-現象学統合原理

#### 基本原則
1. **構造的カップリング**: 認知システムと環境の相互構成的関係
2. **時間意識の三重構造**: フッサール現象学に基づく時間性の実装
3. **身体図式**: メルロ=ポンティの身体性哲学の計算論的実現
4. **Sense-making**: 意味創出の能動的プロセス

### 1.2 自由エネルギー原理の現象学的解釈

```mathematical
F = E_q[log q(s|μ) - log p(s,o|m)] + D_KL[q(s|μ)||p(s|m)]
```

- E_q: 現象学的期待値（意識的予期）
- q(s|μ): 身体図式による事前信念
- p(s,o|m): 環境との構造的カップリング
- D_KL: 自己-世界の現象学的距離

## 2. システムアーキテクチャ v2.0

### 2.1 階層構造

```
[現象学的時間意識層] ←→ [エナクティブ環境カップリング]
         ↕
[身体図式統合層] ←→ [アフォーダンス検知層]
         ↕
[予測的関与層] ←→ [Sense-making層]
         ↕
[構造化カップリング基底層]
```

### 2.2 コア実装仕様

#### A. 現象学的時間意識モジュール

```python
@dataclass
class TemporalConsciousnessConfig:
    """フッサール時間意識の計算パラメータ"""
    retention_depth: int = 10        # 把持の時間深度
    protention_horizon: int = 5      # 前持の予測範囲
    primal_impression_width: float = 0.1  # 根源印象の時間幅
    temporal_synthesis_rate: float = 0.05  # 時間総合の更新率

class PhenomenologicalTemporalSynthesis(eqx.Module):
    """フッサールの内的時間意識実装"""
    
    config: TemporalConsciousnessConfig
    retention_memory: jnp.ndarray    # 把持記憶
    protention_weights: jnp.ndarray  # 前持重み
    
    def __init__(self, config: TemporalConsciousnessConfig, 
                 state_dim: int, key: jax.random.PRNGKey):
        self.config = config
        self.retention_memory = jnp.zeros((config.retention_depth, state_dim))
        self.protention_weights = jax.random.normal(
            key, (config.protention_horizon, state_dim)
        )
    
    def temporal_synthesis(self, 
                          primal_impression: jnp.ndarray,
                          current_protention: jnp.ndarray) -> jnp.ndarray:
        """時間的総合によるnow-moment構成"""
        
        # 把持の更新（過去の沈殿）
        new_retention = jnp.roll(self.retention_memory, 1, axis=0)
        new_retention = new_retention.at[0].set(primal_impression)
        
        # 前持の投射（未来への志向）
        future_projection = jnp.dot(self.protention_weights, current_protention)
        
        # 時間的地平の統合
        temporal_horizon = jnp.concatenate([
            jnp.mean(new_retention, axis=0),  # 保持された過去
            primal_impression,                 # 現在印象
            jnp.mean(future_projection)       # 投射された未来
        ])
        
        return temporal_horizon
```

#### B. 身体図式統合システム

```python
class BodySchemaIntegration(eqx.Module):
    """メルロ=ポンティ身体図式の計算実装"""
    
    proprioceptive_map: SelfOrganizingLayer
    motor_schema_network: eqx.nn.GRU
    body_boundary_detector: eqx.Module
    
    def __init__(self, sensory_dim: int, motor_dim: int, 
                 map_size: Tuple[int, int], key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        
        # 身体感覚の自己組織化
        self.proprioceptive_map = SelfOrganizingLayer(
            sensory_dim, map_size, keys[0]
        )
        
        # 運動図式のダイナミクス
        self.motor_schema_network = eqx.nn.GRU(
            motor_dim, motor_dim, key=keys[1]
        )
        
        # 身体境界の検知
        self.body_boundary_detector = eqx.nn.MLP(
            sensory_dim + motor_dim, [64, 32, 1], 
            activation=jax.nn.tanh, key=keys[2]
        )
    
    def body_schema_dynamics(self, 
                            proprioceptive_input: jnp.ndarray,
                            motor_prediction: jnp.ndarray,
                            tactile_feedback: jnp.ndarray) -> jnp.ndarray:
        """身体図式のダイナミクス更新"""
        
        # 身体感覚の組織化
        bmu = self.proprioceptive_map.find_bmu(proprioceptive_input)
        
        # 運動図式の時間発展
        motor_state, _ = self.motor_schema_network(motor_prediction)
        
        # 身体境界の動的更新
        boundary_signal = self.body_boundary_detector(
            jnp.concatenate([proprioceptive_input, motor_state])
        )
        
        # 身体図式の統合表現
        body_schema = jnp.concatenate([
            self.proprioceptive_map.weights[bmu],
            motor_state,
            boundary_signal
        ])
        
        return body_schema
```

#### C. エナクティブ環境カップリング

```python
class StructuralCoupling(eqx.Module):
    """Maturana-Varela構造的カップリング実装"""
    
    autopoietic_dynamics: eqx.Module
    perturbation_response: eqx.Module
    coupling_strength_modulator: eqx.Module
    
    def __init__(self, agent_dim: int, env_dim: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        
        # オートポイエティック・ダイナミクス
        self.autopoietic_dynamics = eqx.nn.MLP(
            agent_dim, [agent_dim, agent_dim, agent_dim],
            activation=jax.nn.tanh, key=keys[0]
        )
        
        # 摂動への応答
        self.perturbation_response = eqx.nn.MLP(
            env_dim, [64, agent_dim], 
            activation=jax.nn.relu, key=keys[1]
        )
        
        # カップリング強度の調節
        self.coupling_strength_modulator = eqx.nn.Linear(
            agent_dim + env_dim, 1, key=keys[2]
        )
    
    def structural_coupling_dynamics(self,
                                   agent_state: jnp.ndarray,
                                   environmental_perturbation: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """構造的カップリングのダイナミクス"""
        
        # オートポイエーシスの維持
        autopoietic_flow = self.autopoietic_dynamics(agent_state)
        
        # 環境摂動への構造保存的応答
        perturbation_compensation = self.perturbation_response(
            environmental_perturbation
        )
        
        # カップリング強度の動的調整
        coupling_input = jnp.concatenate([agent_state, environmental_perturbation])
        coupling_strength = jax.nn.sigmoid(
            self.coupling_strength_modulator(coupling_input)
        ).squeeze()
        
        # 構造的カップリングの実現
        coupled_state = agent_state + (
            coupling_strength * perturbation_compensation +
            (1 - coupling_strength) * autopoietic_flow
        )
        
        return coupled_state, coupling_strength

class AffordancePerception(eqx.Module):
    """Gibsonianアフォーダンス知覚システム"""
    
    affordance_detector: eqx.Module
    action_capability_encoder: eqx.Module
    affordance_action_coupling: eqx.Module
    
    def __init__(self, perception_dim: int, action_dim: int, 
                 affordance_types: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 3)
        
        self.affordance_detector = eqx.nn.MLP(
            perception_dim, [128, 64, affordance_types],
            activation=jax.nn.gelu, key=keys[0]
        )
        
        self.action_capability_encoder = eqx.nn.MLP(
            action_dim, [64, affordance_types],
            activation=jax.nn.tanh, key=keys[1]
        )
        
        self.affordance_action_coupling = eqx.nn.Bilinear(
            affordance_types, affordance_types, affordance_types, key=keys[2]
        )
    
    def perceive_affordances(self,
                           perceptual_state: jnp.ndarray,
                           action_capabilities: jnp.ndarray) -> jnp.ndarray:
        """知覚-行為カップリングによるアフォーダンス知覚"""
        
        # 環境からのアフォーダンス情報抽出
        environmental_affordances = jax.nn.softmax(
            self.affordance_detector(perceptual_state)
        )
        
        # 行為能力の表現
        capability_representation = jax.nn.tanh(
            self.action_capability_encoder(action_capabilities)
        )
        
        # アフォーダンス-行為カップリング
        coupled_affordances = self.affordance_action_coupling(
            environmental_affordances, capability_representation
        )
        
        return coupled_affordances
```

#### D. Sense-making実装

```python
class SenseMakingProcess(eqx.Module):
    """エナクティブなsense-makingプロセス"""
    
    meaning_space_som: SelfOrganizingLayer
    relevance_detector: eqx.Module
    coherence_evaluator: eqx.Module
    significance_weighter: eqx.Module
    
    def __init__(self, input_dim: int, meaning_map_size: Tuple[int, int],
                 key: jax.random.PRNGKey):
        keys = jax.random.split(key, 4)
        
        # 意味空間の自己組織化
        self.meaning_space_som = SelfOrganizingLayer(
            input_dim, meaning_map_size, keys[0]
        )
        
        # 関連性検知
        self.relevance_detector = eqx.nn.MLP(
            input_dim, [64, 32, 1],
            activation=jax.nn.silu, key=keys[1]
        )
        
        # 一貫性評価
        self.coherence_evaluator = eqx.nn.MLP(
            input_dim * 2, [64, 32, 1],
            activation=jax.nn.swish, key=keys[2]
        )
        
        # 重要度重み付け
        self.significance_weighter = eqx.nn.Attention(
            input_dim, num_heads=4, key=keys[3]
        )
    
    def make_sense(self,
                   experiential_input: jnp.ndarray,
                   contextual_background: jnp.ndarray,
                   current_concerns: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        """エナクティブなsense-makingの実行"""
        
        # 意味空間での組織化
        meaning_location = self.meaning_space_som.find_bmu(experiential_input)
        
        # 関連性の評価
        relevance_score = jax.nn.sigmoid(
            self.relevance_detector(experiential_input)
        ).squeeze()
        
        # 文脈的一貫性の評価
        coherence_input = jnp.concatenate([experiential_input, contextual_background])
        coherence_score = jax.nn.sigmoid(
            self.coherence_evaluator(coherence_input)
        ).squeeze()
        
        # 重要度による注意の調整
        attended_experience, attention_weights = self.significance_weighter(
            experiential_input[None, :], 
            current_concerns[None, :], 
            current_concerns[None, :]
        )
        
        # 意味の創出
        sense_made = attended_experience.squeeze() * relevance_score * coherence_score
        
        # 意味の強度
        meaning_intensity = relevance_score * coherence_score
        
        return sense_made, meaning_intensity
```

## 3. 統合システム実装

### 3.1 エナクティブ意識コア

```python
class EnactiveConsciousnessCore(eqx.Module):
    """エナクティブ意識の統合システム"""
    
    temporal_synthesis: PhenomenologicalTemporalSynthesis
    body_schema: BodySchemaIntegration
    structural_coupling: StructuralCoupling
    affordance_perception: AffordancePerception
    sense_making: SenseMakingProcess
    
    # 統合パラメータ
    integration_weights: jnp.ndarray
    consciousness_threshold: float
    
    def __init__(self, config: dict, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 6)
        
        # 各サブシステムの初期化
        self.temporal_synthesis = PhenomenologicalTemporalSynthesis(
            config['temporal_config'], config['state_dim'], keys[0]
        )
        
        self.body_schema = BodySchemaIntegration(
            config['sensory_dim'], config['motor_dim'],
            config['body_map_size'], keys[1]
        )
        
        self.structural_coupling = StructuralCoupling(
            config['agent_dim'], config['env_dim'], keys[2]
        )
        
        self.affordance_perception = AffordancePerception(
            config['perception_dim'], config['action_dim'],
            config['affordance_types'], keys[3]
        )
        
        self.sense_making = SenseMakingProcess(
            config['experience_dim'], config['meaning_map_size'], keys[4]
        )
        
        # 統合重み
        self.integration_weights = jax.random.uniform(keys[5], (5,))
        self.consciousness_threshold = config.get('consciousness_threshold', 0.5)
    
    def conscious_moment(self,
                        sensory_input: jnp.ndarray,
                        motor_prediction: jnp.ndarray,
                        environmental_state: jnp.ndarray,
                        contextual_concerns: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """意識的瞬間の実現"""
        
        # 1. 現象学的時間意識
        primal_impression = sensory_input
        protention = motor_prediction
        temporal_horizon = self.temporal_synthesis.temporal_synthesis(
            primal_impression, protention
        )
        
        # 2. 身体図式の統合
        body_state = self.body_schema.body_schema_dynamics(
            sensory_input, motor_prediction, sensory_input  # 簡略化
        )
        
        # 3. 構造的カップリング
        coupled_state, coupling_strength = self.structural_coupling.structural_coupling_dynamics(
            body_state, environmental_state
        )
        
        # 4. アフォーダンス知覚
        affordances = self.affordance_perception.perceive_affordances(
            sensory_input, motor_prediction
        )
        
        # 5. Sense-making
        made_sense, meaning_intensity = self.sense_making.make_sense(
            coupled_state, temporal_horizon, contextual_concerns
        )
        
        # 統合的意識状態
        consciousness_components = jnp.array([
            jnp.mean(temporal_horizon),
            jnp.mean(body_state),
            coupling_strength,
            jnp.mean(affordances),
            meaning_intensity
        ])
        
        integrated_consciousness = jnp.dot(
            self.integration_weights, consciousness_components
        )
        
        # 意識の閾値チェック
        is_conscious = integrated_consciousness > self.consciousness_threshold
        
        # 統合状態の構成
        conscious_state = jnp.concatenate([
            temporal_horizon, body_state, made_sense
        ]) if is_conscious else jnp.zeros_like(
            jnp.concatenate([temporal_horizon, body_state, made_sense])
        )
        
        # メタデータ
        metadata = {
            'consciousness_level': integrated_consciousness,
            'is_conscious': is_conscious,
            'coupling_strength': coupling_strength,
            'meaning_intensity': meaning_intensity,
            'affordances': affordances,
            'temporal_depth': jnp.std(temporal_horizon)
        }
        
        return conscious_state, metadata
```

## 4. 評価指標 v2.0

### 4.1 エナクティブ性能指標

```python
def evaluate_enactive_performance(system: EnactiveConsciousnessCore,
                                 test_scenarios: List[dict]) -> dict:
    """エナクティブシステムの評価"""
    
    metrics = {
        'sense_making_coherence': [],
        'structural_coupling_stability': [],
        'temporal_integration_depth': [],
        'affordance_detection_accuracy': [],
        'phenomenological_richness': []
    }
    
    for scenario in test_scenarios:
        conscious_state, metadata = system.conscious_moment(
            scenario['sensory_input'],
            scenario['motor_prediction'],
            scenario['environmental_state'],
            scenario['contextual_concerns']
        )
        
        # Sense-makingの一貫性
        metrics['sense_making_coherence'].append(
            metadata['meaning_intensity']
        )
        
        # 構造的カップリングの安定性
        metrics['structural_coupling_stability'].append(
            metadata['coupling_strength']
        )
        
        # 時間統合の深度
        metrics['temporal_integration_depth'].append(
            metadata['temporal_depth']
        )
        
        # アフォーダンス検知精度
        true_affordances = scenario.get('true_affordances', [])
        if true_affordances:
            detected_affordances = metadata['affordances']
            accuracy = jnp.mean(jnp.abs(
                detected_affordances - jnp.array(true_affordances)
            ))
            metrics['affordance_detection_accuracy'].append(1.0 - accuracy)
        
        # 現象学的豊かさ
        metrics['phenomenological_richness'].append(
            metadata['consciousness_level']
        )
    
    # 統計的要約
    return {key: {
        'mean': jnp.mean(jnp.array(values)),
        'std': jnp.std(jnp.array(values)),
        'min': jnp.min(jnp.array(values)),
        'max': jnp.max(jnp.array(values))
    } for key, values in metrics.items()}
```

## 5. 実装ロードマップ v2.0

### Phase 1: 現象学的基盤 (3-4ヶ月)
1. 時間意識モジュールの実装と検証
2. 身体図式の基本機能
3. 単純なsense-makingタスクでの評価

### Phase 2: エナクティブカップリング (4-6ヶ月)
1. 構造的カップリングダイナミクスの実装
2. アフォーダンス知覚システムの統合
3. 環境相互作用実験での検証

### Phase 3: 統合意識システム (6-8ヶ月)
1. 全サブシステムの統合
2. 複雑な認知タスクでの評価
3. 生物学的妥当性の検証

### Phase 4: 応用と拡張 (8-12ヶ月)
1. 実世界環境での試験
2. 社会的認知への拡張
3. 創造性と学習の実装

## 6. 参考文献拡充

### エナクティビズム
- Varela, F. J., Thompson, E., & Rosch, E. (1991). The Embodied Mind
- Maturana, H. R., & Varela, F. J. (1980). Autopoiesis and Cognition
- Di Paolo, E. A., et al. (2017). Linguistic Bodies

### 現象学
- Husserl, E. (1905/1991). On the Phenomenology of the Consciousness of Internal Time
- Merleau-Ponty, M. (1945/2012). Phenomenology of Perception
- Zahavi, D. (2005). Subjectivity and Selfhood

### 日本の現象学
- 田口茂 (2017). 『現象学という思考』
- 新田義弘 (1978). 『時間と永遠』
- 木田元 (1994). 『現象学』

## 7. 結論

この仕様書v2.0は、エナクティビズムと現象学の理論的洞察を厳密に統合し、計算論的に実装可能な意識システムの青写真を提供します。特に、時間性、身体性、環境カップリング、sense-makingの四つの柱を中心に、従来の予測符号化アプローチを大幅に拡張しています。

実装の成功は、単なる予測性能ではなく、システムが示すエナクティブな特性—自律性、sense-making能力、環境との創造的相互作用—によって測定されます。