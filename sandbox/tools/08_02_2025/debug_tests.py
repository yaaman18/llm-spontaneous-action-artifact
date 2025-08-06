#!/usr/bin/env python3
"""失敗テストのデバッグ"""

from temporal_consciousness import RhythmicMemorySystem, TemporalDistressSystem

# Test 1: 大きなズレは驚きとして体験される
print("=== Test 1: RhythmicMemorySystem ===")
system = RhythmicMemorySystem()
# 安定したリズムを形成
for interval in [1.0, 1.0, 1.0, 1.0]:
    result = system.update_rhythm(interval)
    print(f"Update {interval}: result={result}")

print(f"Internal rhythm: {system.internal_rhythm}")
print(f"History: {system.interval_history}")

# 突然の長い間隔
result = system.update_rhythm(2.0)
print(f"\nUpdate 2.0: result={result}")
print(f"Rhythmic surprise: {result['rhythmic_surprise']}")

# Test 2: 短すぎる間隔で急かされる感覚が生じる
print("\n=== Test 2: TemporalDistressSystem ===")
system2 = TemporalDistressSystem()
result2 = system2.experience_temporal_uncertainty(expected=2.0, actual=1.0)
print(f"Result: {result2}")
print(f"Ratio: {1.0 / 2.0}")
print(f"Expected intensity: {(1.0 / 0.5 - 1.0) * 0.5}")