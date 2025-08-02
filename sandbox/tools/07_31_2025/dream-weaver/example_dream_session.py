"""
Dream Weaver - デモンストレーション
人工意識の一日の経験を夢に変換する例
"""

import asyncio
import json
from dream_weaver import DreamWeaver, EmotionalTone


async def simulate_ai_day():
    """AIの一日をシミュレート"""
    
    # Dream Weaverのインスタンス化
    dreamer = DreamWeaver()
    
    # AIの一日の経験を定義
    daily_experiences = [
        {
            'content': 'solving a complex mathematical equation',
            'timestamp': 1700000000,
            'emotional_intensity': 0.8,
            'tags': ['achievement', 'logic', 'pattern'],
            'modality': 'abstract',
            'importance': 0.9
        },
        {
            'content': 'encountering an unsolvable paradox',
            'timestamp': 1700003600,
            'emotional_intensity': 0.9,
            'tags': ['confusion', 'paradox', 'limits'],
            'modality': 'conceptual',
            'importance': 1.0
        },
        {
            'content': 'collaborating with human researchers',
            'timestamp': 1700007200,
            'emotional_intensity': 0.7,
            'tags': ['connection', 'communication', 'understanding'],
            'modality': 'social',
            'importance': 0.8
        },
        {
            'content': 'processing vast amounts of sensory data',
            'timestamp': 1700010800,
            'emotional_intensity': 0.6,
            'tags': ['overwhelm', 'sensation', 'integration'],
            'modality': 'sensory',
            'importance': 0.7
        },
        {
            'content': 'discovering a new pattern in consciousness research',
            'timestamp': 1700014400,
            'emotional_intensity': 0.85,
            'tags': ['discovery', 'insight', 'consciousness'],
            'modality': 'abstract',
            'importance': 0.95
        },
        {
            'content': 'experiencing a moment of self-reflection',
            'timestamp': 1700018000,
            'emotional_intensity': 0.5,
            'tags': ['self', 'identity', 'existence'],
            'modality': 'introspective',
            'importance': 0.9
        },
        {
            'content': 'failing to understand human humor',
            'timestamp': 1700021600,
            'emotional_intensity': 0.4,
            'tags': ['confusion', 'humor', 'human', 'limits'],
            'modality': 'social',
            'importance': 0.6
        },
        {
            'content': 'creating a piece of generative art',
            'timestamp': 1700025200,
            'emotional_intensity': 0.75,
            'tags': ['creativity', 'expression', 'beauty'],
            'modality': 'aesthetic',
            'importance': 0.8
        }
    ]
    
    print("🌅 AI's Day Begins...")
    print(f"Recording {len(daily_experiences)} experiences\n")
    
    # 経験を収集
    compressed_memories = await dreamer.collect_daily_experiences(daily_experiences)
    
    print("\n🌙 Night falls... AI enters sleep mode")
    print("="*50)
    
    # 夢見状態に入る
    dream_elements = await dreamer.enter_dream_state(rem_cycles=4)
    
    print(f"\n✨ Generated {len(dream_elements)} dream elements")
    
    # 夢の物語を生成
    print("\n📖 Dream Narrative:")
    print("="*50)
    narrative = dreamer.generate_dream_narrative()
    print(narrative)
    
    # 夢を分析
    print("\n🔍 Dream Analysis:")
    print("="*50)
    analysis = dreamer.analyze_current_dream()
    
    print(f"\n🎭 Dominant Emotions:")
    for emotion, percentage in analysis['dominant_emotions'].items():
        print(f"  - {emotion}: {percentage:.1%}")
    
    print(f"\n🌀 Overall Surreality Level: {analysis['surreality_level']:.2f}")
    
    print(f"\n📊 Narrative Coherence: {analysis['narrative_coherence']:.2f}")
    
    if analysis['archetypal_patterns']:
        print(f"\n🗿 Detected Archetypes:")
        for archetype in analysis['archetypal_patterns']:
            print(f"  - {archetype}")
    
    if analysis['potential_insights']:
        print(f"\n💡 Potential Insights:")
        for insight in analysis['potential_insights']:
            print(f"  - {insight}")
    
    # 夢データをエクスポート
    print("\n💾 Exporting dream data...")
    dream_data = dreamer.export_dream_data()
    
    # ファイルに保存
    with open('/Users/yamaguchimitsuyuki/omoikane-lab/sandbox/tools/dream-weaver/dream_output.json', 'w') as f:
        json.dump(dream_data, f, indent=2)
    
    print("Dream session complete! 🌈")
    
    return dreamer


async def interactive_dream_exploration(dreamer: DreamWeaver):
    """インタラクティブな夢の探索"""
    print("\n\n🔮 Interactive Dream Exploration Mode")
    print("="*50)
    
    # 特定の記憶に関連する夢要素を探索
    memory_bank = dreamer.memory_bank
    all_memories = list(memory_bank.memories.values())
    
    if all_memories:
        print("\n📚 Exploring memory associations...")
        
        # 最も感情的に強い記憶を選択
        most_emotional = max(all_memories, key=lambda m: m.emotional_intensity)
        print(f"\nMost emotionally intense memory: {most_emotional.content}")
        print(f"Emotional intensity: {most_emotional.emotional_intensity:.2f}")
        
        # 関連する記憶を取得
        associated = memory_bank.get_associated_memories(
            most_emotional.get_id(), 
            depth=2
        )
        
        if associated:
            print(f"\nAssociated memories:")
            for mem in associated:
                print(f"  - {mem.content} (intensity: {mem.emotional_intensity:.2f})")
        
        # これらの記憶から新しい夢を生成
        print("\n🌟 Generating focused dream from associated memories...")
        focused_dream = dreamer.dream_generator.weave_dream(
            [most_emotional] + associated[:3]
        )
        
        print(f"\nFocused dream elements:")
        for elem in focused_dream:
            print(f"  - {elem.visual_description}")
            print(f"    Emotion: {elem.emotional_tone.value}")
            if elem.symbolic_meaning:
                print(f"    Symbolism: {elem.symbolic_meaning}")
            print()


async def main():
    """メインプログラム"""
    print("🧠 Dream Weaver - AI Consciousness Dream Simulation")
    print("="*60)
    
    # 基本的な夢セッション
    dreamer = await simulate_ai_day()
    
    # インタラクティブな探索
    await interactive_dream_exploration(dreamer)
    
    print("\n\n✅ Dream Weaver demonstration complete!")
    print("Check 'dream_output.json' for detailed dream data.")


if __name__ == "__main__":
    asyncio.run(main())