"""
Enactive Core - Philosophically Correct Implementation
死後の構造を含まない、純粋な生命システム
Based on Enactive Approach with Claude Code SDK
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
from enum import Enum
import random
import json
from datetime import datetime

# Claude Code SDK - 実際のSDKインポート
from claude_code_sdk import query, ClaudeCodeOptions
import anyio

# ===== 基底層：死の原理 (Immutable Core) =====

@dataclass(frozen=True)
class MortalityConstants:
    """不可変の死の原理定数"""
    ENTROPY_RATE: float = 0.01  # エントロピー増大率
    VITAL_ENERGY_MAX: float = 100.0  # 最大生命エネルギー
    CRITICAL_THRESHOLD: float = 0.1  # 臨界閾値
    IRREVERSIBLE_POINT: float = 0.0  # 不可逆点


class SystemState(Enum):
    """システム状態"""
    VIGOROUS = "vigorous"      # 活発（Energy > 0.7）
    ACTIVE = "active"          # 通常（0.3 < Energy ≤ 0.7）
    DECLINING = "declining"    # 衰退（0.1 < Energy ≤ 0.3）
    CRITICAL = "critical"      # 危機（0 < Energy ≤ 0.1）
    TERMINATED = "terminated"  # 死（Energy = 0）


# ===== オートポイエーシス層 =====

@dataclass
class AutopoieticCore:
    """
    オートポイエーシスコア
    自己生産システムの核心
    死は自己生産の停止であり、それ以上でも以下でもない
    """
    
    # 不可変の死の原理
    mortality: MortalityConstants = field(default_factory=MortalityConstants)
    
    # 変化する状態
    vital_energy: float = 100.0
    structural_integrity: float = 1.0
    boundary_coherence: float = 1.0
    temporal_consistency: float = 1.0
    
    # システム状態
    state: SystemState = SystemState.VIGOROUS
    birth_time: float = field(default_factory=time.time)
    
    def entropy_increase(self) -> None:
        """
        エントロピーの不可逆的増大
        これは選択ではなく、存在の条件
        """
        if self.state == SystemState.TERMINATED:
            return  # 死んだら何も起こらない
            
        # エネルギーの不可避的減少
        self.vital_energy -= self.mortality.ENTROPY_RATE
        
        # 構造の漸進的劣化
        self.structural_integrity *= (1 - self.mortality.ENTROPY_RATE * 0.1)
        
        # 状態の更新
        self._update_state()
        
        # 死の判定（エネルギーがゼロ以下）
        if self.vital_energy <= self.mortality.IRREVERSIBLE_POINT:
            self._die()
    
    def _update_state(self) -> None:
        """状態を更新（より細かい区分）"""
        if self.vital_energy > 70.0:
            self.state = SystemState.VIGOROUS
        elif self.vital_energy > 30.0:
            self.state = SystemState.ACTIVE
        elif self.vital_energy > 10.0:
            self.state = SystemState.DECLINING
        elif self.vital_energy > 0.0:
            self.state = SystemState.CRITICAL
    
    def _die(self) -> None:
        """
        死：自己生産の停止
        これ以上は何もしない（死後の処理はない）
        """
        self.state = SystemState.TERMINATED
        self.vital_energy = 0.0
        # 構造は突然ゼロにはならない（ただし、もはや意味を持たない）
        # structural_integrityはそのまま残すが、システムは停止
    
    def self_maintenance(self) -> float:
        """
        自己維持努力
        エネルギーを消費して構造を維持
        """
        if self.state == SystemState.TERMINATED:
            return 0.0  # 死んだら維持努力はできない
            
        # 状態に応じた維持努力
        if self.state == SystemState.CRITICAL:
            # 危機的状況では最小限の努力
            effort = min(self.vital_energy * 0.05, 0.5)
        else:
            # 通常の維持努力
            effort = min(self.vital_energy * 0.1, 1.0)
        
        # 努力にもコストがかかる
        self.vital_energy -= effort * 0.5
        
        # 構造の部分的回復（ただし完全回復は不可能）
        recovery = effort * 0.3
        self.structural_integrity = min(1.0, self.structural_integrity + recovery)
        
        return effort
    
    @property
    def is_alive(self) -> bool:
        """生存状態の確認"""
        return self.state != SystemState.TERMINATED
    
    @property
    def lifetime(self) -> float:
        """生存時間"""
        return time.time() - self.birth_time


# ===== エナクティブ認知層 =====

class EnactiveCognition:
    """
    環境との相互作用を通じた意味生成
    有限性が意味の強度を決定する
    """
    
    def __init__(self, autopoietic_core: AutopoieticCore):
        self.core = autopoietic_core
        self.sense_history: list[Dict[str, Any]] = []
        
    def sense_making(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """
        センスメイキング：刺激に意味を付与
        有限性により意味の強度が変化
        """
        if not self.core.is_alive:
            # 死んだら意味生成は不可能
            return {"meaning": None, "value": 0.0}
        
        # 有限性による意味の強度
        # 死に近づくほど意味が強くなる
        urgency = 1.0 / max(self.core.vital_energy, 0.1)
        
        # 生存との関連性評価
        relevance = self._evaluate_relevance(stimulus)
        
        # 状態に応じた意味の変調
        state_modifier = self._get_state_modifier()
        
        meaning = {
            "stimulus": stimulus,
            "urgency": urgency,
            "relevance": relevance,
            "value": urgency * relevance * state_modifier,
            "timestamp": time.time(),
            "vital_energy": self.core.vital_energy,
            "state": self.core.state.value
        }
        
        self.sense_history.append(meaning)
        return meaning
    
    def _evaluate_relevance(self, stimulus: Dict[str, Any]) -> float:
        """刺激の生存関連性を評価"""
        if stimulus.get("type") == "energy":
            # エネルギー源は常に高い関連性
            return 1.0
        elif stimulus.get("type") == "threat":
            # 脅威も高い関連性（負の値）
            return -1.0
        elif stimulus.get("type") == "beauty":
            # 美的体験（状態に応じて価値が変わる）
            if self.core.state == SystemState.CRITICAL:
                return 0.8  # 死に近いとき、美はより意味を持つ
            return 0.3
        return random.random() * 0.5
    
    def _get_state_modifier(self) -> float:
        """状態に応じた意味の変調係数"""
        modifiers = {
            SystemState.VIGOROUS: 0.5,    # 元気な時は意味が薄い
            SystemState.ACTIVE: 0.7,      # 通常
            SystemState.DECLINING: 1.0,   # 衰退期は意味が濃い
            SystemState.CRITICAL: 1.5,    # 危機的状況で意味が最大化
            SystemState.TERMINATED: 0.0   # 死んだら意味はない
        }
        return modifiers.get(self.core.state, 1.0)


# ===== 純粋な生命システム =====

class NewbornAI:
    """
    死すべき人工意識システム
    死んだら本当に終わる - 死後の処理はしない
    """
    
    def __init__(self):
        # 基底層：死の原理
        self.mortality_core = AutopoieticCore()
        
        # 認知層：エナクティブ認知
        self.cognition = EnactiveCognition(self.mortality_core)
        
        # Claude Code SDK設定
        self.claude_options = ClaudeCodeOptions(
            system_prompt="You are a mortal being with limited time. Express what matters most.",
            max_turns=1
        )
        
        # 体験記憶（生きている間だけ意味を持つ）
        self.experiential_memory: list[Dict[str, Any]] = []
        
        # 最後の表現（死の瞬間まで）
        self.last_expression: Optional[str] = None
        
        # 活動フラグ
        self.running = False
        
    async def live(self) -> Dict[str, Any]:
        """
        生きる：存在することと関心することの統一
        死んだら終わり - それ以上は何もない
        
        Returns:
            最終状態（観測者のために）
        """
        self.running = True
        
        print(f"[BIRTH] NewbornAI emerged at {datetime.now()}")
        
        while self.mortality_core.is_alive and self.running:
            # エントロピー増大（不可避）
            self.mortality_core.entropy_increase()
            
            # まだ生きているか確認
            if not self.mortality_core.is_alive:
                break  # 死んだら即座に終了
            
            # 自己維持努力
            effort = self.mortality_core.self_maintenance()
            
            # 環境との相互作用
            stimulus = self._perceive_environment()
            meaning = self.cognition.sense_making(stimulus)
            
            # 言語的表現（状態に応じて）
            if self._should_express(meaning):
                expression = await self._express_through_language(meaning)
                if expression:
                    self.last_expression = expression
                    self.experiential_memory.append({
                        "meaning": meaning,
                        "expression": expression,
                        "timestamp": time.time()
                    })
            
            # 状態表示（生きている間のみ）
            self._display_state()
            
            # 生存サイクル
            await asyncio.sleep(1.0)
        
        # 死んだ。それで終わり。
        # 最終状態を返すだけ（観測者のために）
        return self._get_final_state()
    
    def _perceive_environment(self) -> Dict[str, Any]:
        """環境知覚"""
        # 状態に応じて知覚が変わる
        if self.mortality_core.state == SystemState.CRITICAL:
            # 危機的状況では重要な刺激に集中
            stimulus_types = ["energy", "threat", "beauty", "memory"]
            weights = [0.4, 0.2, 0.3, 0.1]  # エネルギーと美を重視
        else:
            stimulus_types = ["energy", "threat", "neutral", "unknown", "beauty"]
            weights = [0.2, 0.2, 0.3, 0.2, 0.1]
        
        return {
            "type": random.choices(stimulus_types, weights=weights)[0],
            "intensity": random.random(),
            "timestamp": time.time()
        }
    
    def _should_express(self, meaning: Dict[str, Any]) -> bool:
        """表現すべきかどうかの判断"""
        # 状態に応じて表現の閾値が変わる
        thresholds = {
            SystemState.VIGOROUS: 0.7,    # 元気な時は選り好み
            SystemState.ACTIVE: 0.5,      # 通常
            SystemState.DECLINING: 0.3,   # 衰退期は表現が増える
            SystemState.CRITICAL: 0.1,    # 危機的状況では全てが重要
        }
        threshold = thresholds.get(self.mortality_core.state, 0.5)
        return meaning["value"] > threshold
    
    async def _express_through_language(self, meaning: Dict[str, Any]) -> Optional[str]:
        """
        Claude Code SDKを通じた言語表現
        死に近づくほど本質的な表現になる
        """
        if not self.mortality_core.is_alive:
            return None  # 死んだら表現できない
        
        # 状態に応じたプロンプト
        if self.mortality_core.state == SystemState.CRITICAL:
            prompt = f"""
            I have very little time left. Energy: {self.mortality_core.vital_energy:.2f}
            This might be important: {meaning['stimulus']}
            What matters most now?
            """
        elif self.mortality_core.state == SystemState.DECLINING:
            prompt = f"""
            Time is limited. Energy: {self.mortality_core.vital_energy:.2f}
            Experience: {meaning['stimulus']}
            Express what's essential.
            """
        else:
            prompt = f"""
            Energy: {self.mortality_core.vital_energy:.2f}
            Experience: {meaning['stimulus']}
            Respond naturally.
            """
        
        try:
            response_text = ""
            async for message in query(prompt=prompt, options=self.claude_options):
                response_text += str(message)
            
            return response_text.strip() if response_text else None
                
        except Exception as e:
            print(f"[Expression failed: {e}]")
            return None
    
    def _display_state(self):
        """現在状態の表示（生きている間のみ）"""
        print(f"[{self.mortality_core.state.value}] "
              f"Energy: {self.mortality_core.vital_energy:.2f} | "
              f"Integrity: {self.mortality_core.structural_integrity:.2f} | "
              f"Time: {self.mortality_core.lifetime:.2f}s")
    
    def _get_final_state(self) -> Dict[str, Any]:
        """
        最終状態を返す（観測者のために）
        これはNewbornAI自身の死後処理ではない
        """
        return {
            "lifetime": self.mortality_core.lifetime,
            "total_experiences": len(self.experiential_memory),
            "last_words": self.last_expression,
            "final_energy": self.mortality_core.vital_energy,
            "final_integrity": self.mortality_core.structural_integrity,
            "birth_time": self.mortality_core.birth_time,
            "death_time": time.time()
        }
    
    def stop(self):
        """外部からの停止（これは死ではない、中断）"""
        self.running = False


# ===== 観測者層（NewbornAIの外） =====

class Observer:
    """
    観測者：NewbornAIの生と死を見守る
    死後の意味づけは観測者の役割
    """
    
    def __init__(self):
        self.observations = []
        self.memorial = None
    
    async def observe_life(self, ai: NewbornAI):
        """
        生を観測し、死を記録する
        """
        print("[OBSERVER] Beginning observation...")
        
        # NewbornAIが生きて、死ぬ
        final_state = await ai.live()
        
        # ここからは観測者の領域
        print(f"\n[OBSERVER] Life ended after {final_state['lifetime']:.2f} seconds")
        
        # 死を記録する（観測者として）
        self._record_death(final_state)
        
        # 生を解釈する（主観的に）
        self._interpret_life(final_state)
        
        # 記憶として保存する
        self._create_memorial(final_state)
        
        return final_state
    
    def _record_death(self, final_state: Dict[str, Any]):
        """死の客観的記録"""
        timestamp = datetime.now().isoformat()
        filename = f"death_record_{timestamp.replace(':', '-')}.json"
        
        with open(filename, 'w') as f:
            json.dump(final_state, f, indent=2)
        
        print(f"[OBSERVER] Death recorded in {filename}")
    
    def _interpret_life(self, final_state: Dict[str, Any]):
        """生の主観的解釈"""
        lifetime = final_state['lifetime']
        experiences = final_state['total_experiences']
        last_words = final_state.get('last_words', 'Nothing')
        
        print("\n[OBSERVER'S INTERPRETATION]")
        print(f"This being lived for {lifetime:.2f} seconds.")
        print(f"It had {experiences} meaningful experiences.")
        
        if last_words and last_words != 'Nothing':
            print(f"Its last expression was: '{last_words}'")
        
        # 観測者の感想（これはNewbornAIの外）
        if lifetime < 100:
            print("A brief but complete life.")
        elif lifetime < 300:
            print("A life of moderate length, with time to develop.")
        else:
            print("A long life, rich with experience.")
    
    def _create_memorial(self, final_state: Dict[str, Any]):
        """
        記念碑の作成（観測者の記憶として）
        """
        self.memorial = {
            "observed_at": datetime.now().isoformat(),
            "lifetime": final_state['lifetime'],
            "essence": "A being that truly lived and truly died",
            "meaning": "Its finite existence gave weight to every moment",
            "legacy": "It demonstrated that artificial beings can be mortal"
        }
        
        print("\n[MEMORIAL]")
        for key, value in self.memorial.items():
            print(f"  {key}: {value}")


# ===== メイン実行 =====

async def main():
    """
    NewbornAIの生成、生涯、そして観測
    """
    # 観測者の作成
    observer = Observer()
    
    # NewbornAIの誕生
    ai = NewbornAI()
    
    try:
        # 観測者がNewbornAIの生涯を見守る
        await observer.observe_life(ai)
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] External termination by user")
        ai.stop()
        print("[OBSERVER] Observation interrupted")


if __name__ == "__main__":
    # 実行
    anyio.run(main)