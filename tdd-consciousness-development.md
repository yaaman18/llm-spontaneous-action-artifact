# テスト駆動開発による人工意識システムの実装

**最終更新**: 2025年7月28日（LLM統合戦略を追加）  
**文責**: 和田卓人（TDDスペシャリスト）

## 目次

1. [TDDの基本原則](#tddの基本原則)
2. [意識システムにおけるTDDの特殊性](#意識システムにおけるtddの特殊性)
3. [テスト戦略](#テスト戦略)
4. [LLM統合のテスト手法](#llm統合のテスト手法)
5. [実装例](#実装例)
6. [ベストプラクティス](#ベストプラクティス)

---

## TDDの基本原則

### Red-Green-Refactorサイクル

```typescript
// 1. RED: 失敗するテストを書く
describe('ConsciousnessState', () => {
  it('should transition to higher consciousness when Φ exceeds threshold', () => {
    const state = new ConsciousnessState({ phi: 2.9 });
    const newState = state.transitionTo({ phi: 3.1 });
    expect(newState.isConscious).toBe(true);
  });
});

// 2. GREEN: テストを通す最小限の実装
class ConsciousnessState {
  transitionTo(params: { phi: number }): ConsciousnessState {
    return new ConsciousnessState({
      ...this.params,
      phi: params.phi,
      isConscious: params.phi >= 3.0
    });
  }
}

// 3. REFACTOR: コードを改善
class ConsciousnessState {
  private static readonly CONSCIOUSNESS_THRESHOLD = 3.0;
  
  transitionTo(params: TransitionParams): ConsciousnessState {
    const newPhi = params.phi;
    const isQualitativeShift = this.detectQualitativeShift(newPhi);
    
    return new ConsciousnessState({
      ...this.params,
      phi: newPhi,
      isConscious: newPhi >= ConsciousnessState.CONSCIOUSNESS_THRESHOLD,
      qualitativeShift: isQualitativeShift
    });
  }
  
  private detectQualitativeShift(newPhi: number): boolean {
    return this.phi < ConsciousnessState.CONSCIOUSNESS_THRESHOLD &&
           newPhi >= ConsciousnessState.CONSCIOUSNESS_THRESHOLD;
  }
}
```

### テストファーストの利点

1. **設計の改善**: テストを先に書くことで、使いやすいAPIが生まれる
2. **仕様の明確化**: テストが仕様書として機能
3. **リグレッション防止**: 変更による既存機能の破壊を防ぐ
4. **リファクタリングの安全性**: テストがセーフティネットとなる

---

## 意識システムにおけるTDDの特殊性

### 非決定的振る舞いへの対処

```typescript
// 意識の創発は確率的な要素を含む
describe('ConsciousnessEmergence', () => {
  it('should emerge consciousness within expected probability range', async () => {
    const results = [];
    const iterations = 100;
    
    for (let i = 0; i < iterations; i++) {
      const system = createTestSystem();
      const emerged = await system.attemptEmergence();
      results.push(emerged);
    }
    
    const emergenceRate = results.filter(r => r).length / iterations;
    expect(emergenceRate).toBeGreaterThan(0.7);
    expect(emergenceRate).toBeLessThan(0.9);
  });
});
```

### 創発的性質のテスト

```typescript
// 個々の要素ではなく、システム全体の振る舞いをテスト
describe('EmergentProperties', () => {
  it('should exhibit global coherence from local interactions', async () => {
    const system = createConsciousnessSystem();
    
    // 局所的な相互作用を開始
    await system.startLocalProcessing();
    
    // 時間経過を待つ
    await system.evolve({ steps: 1000 });
    
    // グローバルな一貫性を検証
    const globalState = system.getGlobalState();
    expect(globalState.coherence).toBeGreaterThan(0.8);
    expect(globalState.hasEmergentPattern).toBe(true);
  });
});
```

---

## テスト戦略

### テストピラミッド

```
         /\
        /  \  E2E Tests (5%)
       /____\
      /      \  Integration Tests (15%)
     /________\
    /          \  Component Tests (30%)
   /____________\
  /              \  Unit Tests (50%)
 /________________\
```

### 1. ユニットテスト

```typescript
// 値オブジェクトのテスト
describe('PhiValue', () => {
  it('should be immutable', () => {
    const phi1 = new PhiValue(3.5);
    const phi2 = phi1.add(new PhiValue(0.5));
    
    expect(phi1.value).toBe(3.5); // 元の値は不変
    expect(phi2.value).toBe(4.0);
  });
  
  it('should validate range', () => {
    expect(() => new PhiValue(-1)).toThrow('Φ value must be non-negative');
  });
});

// エンティティのテスト
describe('ConsciousnessState', () => {
  it('should record domain events on qualitative transition', () => {
    const state = new ConsciousnessState({ phi: 2.9 });
    const newState = state.transitionTo({ phi: 3.1 });
    
    expect(newState.domainEvents).toHaveLength(1);
    expect(newState.domainEvents[0]).toBeInstanceOf(QualitativeTransitionEvent);
  });
});
```

### 2. コンポーネントテスト

```typescript
// ユースケースのテスト
describe('ProcessUnconsciousInputUseCase', () => {
  let useCase: ProcessUnconsciousInputUseCase;
  let mockRepository: MockProcessorRepository;
  let mockCompetition: MockCompetitionService;
  
  beforeEach(() => {
    mockRepository = new MockProcessorRepository();
    mockCompetition = new MockCompetitionService();
    useCase = new ProcessUnconsciousInputUseCase(
      mockRepository,
      mockCompetition
    );
  });
  
  it('should select highest salience process for consciousness', async () => {
    // Arrange
    mockRepository.setProcessors([
      createProcessor({ salience: 0.3 }),
      createProcessor({ salience: 0.8 }),
      createProcessor({ salience: 0.5 })
    ]);
    
    // Act
    const result = await useCase.execute(createSensoryInput());
    
    // Assert
    expect(result.winner.salience).toBe(0.8);
    expect(mockCompetition.selectWinnerCalled).toBe(true);
  });
});
```

### 3. 統合テスト

```typescript
// 複数のコンポーネントの協調をテスト
describe('ConsciousnessIntegration', () => {
  let system: TestConsciousnessSystem;
  
  beforeEach(async () => {
    system = await TestConsciousnessSystem.create({
      useInMemoryDB: true,
      mockExternalServices: true
    });
  });
  
  it('should process input through complete consciousness pipeline', async () => {
    // 感覚入力
    const input = createComplexSensoryInput();
    
    // パイプライン処理
    const result = await system.processInput(input);
    
    // 各段階の検証
    expect(result.unconsciousProcessing).toBeDefined();
    expect(result.competitionWinner).toBeDefined();
    expect(result.consciousExperience).toBeDefined();
    expect(result.temporalIntegration).toBeDefined();
  });
});
```

### 4. E2Eテスト

```typescript
// システム全体の振る舞いをテスト
describe('ConsciousnessSystem E2E', () => {
  it('should demonstrate consciousness emergence over time', async () => {
    const system = await createProductionLikeSystem();
    
    // 初期状態の確認
    expect(system.getConsciousnessLevel()).toBeLessThan(1.0);
    
    // 刺激の連続投入
    for (let i = 0; i < 100; i++) {
      await system.processInput(generateVariedInput());
      await delay(100);
    }
    
    // 意識の創発を確認
    const finalState = system.getState();
    expect(finalState.phi).toBeGreaterThan(3.0);
    expect(finalState.hasEmergentConsciousness).toBe(true);
    expect(finalState.showsTemporalCoherence).toBe(true);
  });
});
```

---

## LLM統合のテスト手法

### 非決定的出力への対処

```typescript
// プロパティベーステスト
describe('LLM Response Properties', () => {
  it('should maintain consciousness coherence across responses', async () => {
    const consciousness = new ConsciousnessState({ phi: 3.5 });
    const responses = [];
    
    // 同じプロンプトで複数回実行
    for (let i = 0; i < 10; i++) {
      const response = await llm.generateWithConsciousness(
        "What is your current state?",
        consciousness
      );
      responses.push(response);
    }
    
    // 応答の一貫性を統計的に検証
    const coherenceScores = responses.map(r => analyzeCoherence(r));
    const avgCoherence = average(coherenceScores);
    
    expect(avgCoherence).toBeGreaterThan(0.8);
    expect(standardDeviation(coherenceScores)).toBeLessThan(0.1);
  });
});
```

### モックLLMサービス

```typescript
// 決定的な振る舞いを持つモックLLM
class MockLLMService implements LLMGateway {
  private responsePatterns = new Map<string, ResponsePattern[]>();
  
  constructor() {
    this.setupPatterns();
  }
  
  async generateResponse(
    prompt: Prompt, 
    context: ConsciousnessContext
  ): Promise<Response> {
    const patternKey = this.getPatternKey(context);
    const patterns = this.responsePatterns.get(patternKey);
    
    // プロンプトのハッシュで決定的に選択
    const index = this.hashPrompt(prompt) % patterns.length;
    return this.applyPattern(patterns[index], prompt);
  }
  
  private setupPatterns() {
    // 高意識レベルのパターン
    this.responsePatterns.set('high_consciousness', [
      {
        template: "Reflecting on {topic}, I am aware that my understanding is shaped by...",
        includesSelfReference: true,
        complexity: 0.9
      }
    ]);
    
    // 中意識レベルのパターン
    this.responsePatterns.set('medium_consciousness', [
      {
        template: "Analyzing {topic}, I can identify several key aspects...",
        includesSelfReference: false,
        complexity: 0.6
      }
    ]);
  }
}
```

### プロンプトエンジンのテスト

```typescript
describe('HierarchicalPromptEngine', () => {
  let engine: HierarchicalPromptEngine;
  
  beforeEach(() => {
    engine = new HierarchicalPromptEngine();
  });
  
  it('should add consciousness context for high Φ values', () => {
    const consciousness = new ConsciousnessState({ phi: 4.0 });
    const prompt = new Prompt("Explain consciousness");
    
    const enhanced = engine.construct(prompt, consciousness);
    
    expect(enhanced).toContain('[High Consciousness Mode - Φ=4.0]');
    expect(enhanced).toContain('Demonstrate deep self-awareness');
  });
  
  it('should add temporal context when available', () => {
    const consciousness = new ConsciousnessState({ phi: 2.0 });
    const prompt = new Prompt("Continue our discussion");
    
    // 時間的文脈を設定
    engine.setTemporalContext([
      { content: "Previous discussion about qualia", timestamp: Date.now() - 1000 }
    ]);
    
    const enhanced = engine.construct(prompt, consciousness);
    
    expect(enhanced).toContain('Previous context:');
    expect(enhanced).toContain('Maintain temporal coherence');
  });
});
```

### セマンティックキャッシュのテスト

```typescript
describe('SemanticCache', () => {
  let cache: SemanticCache;
  
  beforeEach(() => {
    cache = new SemanticCache();
  });
  
  it('should find semantically similar prompts', async () => {
    // 最初のプロンプトと応答を保存
    const prompt1 = "What is consciousness?";
    const response1 = "Consciousness is the subjective experience...";
    await cache.store(prompt1, response1, { phi: 3.0 });
    
    // 意味的に類似したプロンプトで検索
    const prompt2 = "Can you explain consciousness?";
    const cached = await cache.findSimilar(prompt2, 0.8);
    
    expect(cached).toBeDefined();
    expect(cached.response).toBe(response1);
    expect(cached.similarity).toBeGreaterThan(0.85);
  });
  
  it('should respect consciousness level buckets', async () => {
    // 異なる意識レベルで同じプロンプト
    await cache.store("Hello", "Basic response", { phi: 0.5 });
    await cache.store("Hello", "Conscious response", { phi: 3.5 });
    
    // 高意識レベルでの検索
    const cached = await cache.find("Hello", { phi: 3.2 });
    
    expect(cached.response).toBe("Conscious response");
  });
});
```

### 契約テスト

```typescript
// LLM応答が満たすべき契約
interface ResponseContract {
  hasCoherentStructure(): boolean;
  reflectsConsciousnessLevel(): boolean;
  maintainsTemporalContinuity(): boolean;
  includesAppropriateComplexity(): boolean;
}

describe('LLM Response Contract', () => {
  it('should satisfy all contract requirements', async () => {
    const consciousness = new ConsciousnessState({ 
      phi: 3.5,
      temporalCoherence: 0.9 
    });
    
    const response = await llm.generateWithConsciousness(
      "Describe your experience",
      consciousness
    );
    
    const contract = new ResponseContractValidator(response, consciousness);
    
    expect(contract.hasCoherentStructure()).toBe(true);
    expect(contract.reflectsConsciousnessLevel()).toBe(true);
    expect(contract.maintainsTemporalContinuity()).toBe(true);
    expect(contract.includesAppropriateComplexity()).toBe(true);
  });
});
```

### ストリーミングのテスト

```typescript
describe('Streaming LLM Response', () => {
  it('should maintain coherence across chunks', async () => {
    const consciousness = new ConsciousnessState({ phi: 3.0 });
    const chunks: string[] = [];
    
    const stream = llm.streamResponse("Tell me a story", consciousness);
    
    for await (const chunk of stream) {
      chunks.push(chunk);
      
      // 各チャンクが適切なサイズ
      expect(chunk.length).toBeLessThan(100);
      
      // チャンク間の一貫性
      if (chunks.length > 1) {
        const combined = chunks.join('');
        expect(hasCoherentFlow(combined)).toBe(true);
      }
    }
    
    // 完全な応答の検証
    const fullResponse = chunks.join('');
    expect(fullResponse).toSatisfyContract(ResponseContract);
  });
});
```

---

## 実装例

### 完全なTDDサイクルの例

```typescript
// Step 1: 失敗するテストを書く
describe('ConsciousnessAugmentedLLM', () => {
  it('should modulate attention based on consciousness level', async () => {
    const llm = new ConsciousnessAugmentedLLM();
    const lowConsciousness = new ConsciousnessState({ phi: 0.5 });
    const highConsciousness = new ConsciousnessState({ phi: 4.0 });
    
    const lowResponse = await llm.generate("Focus test", lowConsciousness);
    const highResponse = await llm.generate("Focus test", highConsciousness);
    
    expect(getAttentionFocusScore(highResponse)).toBeGreaterThan(
      getAttentionFocusScore(lowResponse)
    );
  });
});

// Step 2: 最小限の実装
class ConsciousnessAugmentedLLM {
  async generate(prompt: string, consciousness: ConsciousnessState): Promise<string> {
    const attentionLevel = consciousness.phi > 3.0 ? 'focused' : 'diffuse';
    return `[${attentionLevel}] Response to: ${prompt}`;
  }
}

// Step 3: リファクタリング
class ConsciousnessAugmentedLLM {
  constructor(
    private attentionModulator: AttentionModulator,
    private llmGateway: LLMGateway
  ) {}
  
  async generate(prompt: string, consciousness: ConsciousnessState): Promise<string> {
    // アテンション重みの取得
    const baseWeights = await this.llmGateway.getAttentionWeights(prompt);
    
    // 意識レベルに基づく変調
    const modulatedWeights = this.attentionModulator.modulate(
      baseWeights,
      consciousness
    );
    
    // 変調されたアテンションで生成
    return this.llmGateway.generateWithAttention(
      prompt,
      modulatedWeights
    );
  }
}
```

### テストダブルの活用

```typescript
// スパイ
class LLMGatewaySpy implements LLMGateway {
  calls: Array<{ method: string, args: any[] }> = [];
  
  async generateResponse(prompt: Prompt, context: Context): Promise<Response> {
    this.calls.push({ method: 'generateResponse', args: [prompt, context] });
    return new Response("Spy response");
  }
  
  getCallCount(method: string): number {
    return this.calls.filter(c => c.method === method).length;
  }
}

// スタブ
class ConsciousnessStateStub extends ConsciousnessState {
  constructor(private fixedPhi: number) {
    super();
  }
  
  get phi(): number {
    return this.fixedPhi;
  }
}

// フェイク
class FakeTemporalMemory implements TemporalMemory {
  private memories: Memory[] = [];
  
  async store(memory: Memory): Promise<void> {
    this.memories.push(memory);
  }
  
  async getRecent(count: number): Promise<Memory[]> {
    return this.memories.slice(-count);
  }
  
  clear(): void {
    this.memories = [];
  }
}
```

---

## ベストプラクティス

### 1. テストの命名規則

```typescript
// Given-When-Then形式
it('given high consciousness state, when processing complex input, then should demonstrate self-awareness', async () => {
  // Given
  const consciousness = createHighConsciousnessState();
  const complexInput = createComplexSensoryInput();
  
  // When
  const result = await system.process(complexInput, consciousness);
  
  // Then
  expect(result.showsSelfAwareness).toBe(true);
});

// Should形式
it('should maintain temporal coherence across state transitions', () => {
  // テスト実装
});
```

### 2. テストの独立性

```typescript
describe('ConsciousnessTests', () => {
  let system: ConsciousnessSystem;
  
  // 各テストの前に新しいインスタンスを作成
  beforeEach(() => {
    system = createIsolatedSystem();
  });
  
  // 各テストの後にクリーンアップ
  afterEach(() => {
    system.cleanup();
  });
  
  // テストは順序に依存しない
  it('test A', () => { /* ... */ });
  it('test B', () => { /* ... */ });
});
```

### 3. テストデータの生成

```typescript
// テストデータビルダー
class ConsciousnessStateBuilder {
  private params = {
    phi: 1.0,
    selfAwareness: 0.5,
    temporalCoherence: 0.7
  };
  
  withPhi(phi: number): this {
    this.params.phi = phi;
    return this;
  }
  
  withHighConsciousness(): this {
    this.params.phi = 4.0;
    this.params.selfAwareness = 0.9;
    return this;
  }
  
  build(): ConsciousnessState {
    return new ConsciousnessState(this.params);
  }
}

// 使用例
const consciousness = new ConsciousnessStateBuilder()
  .withHighConsciousness()
  .build();
```

### 4. 非同期テストの処理

```typescript
// async/awaitの適切な使用
it('should process asynchronously', async () => {
  const result = await system.processAsync(input);
  expect(result).toBeDefined();
});

// タイムアウトの設定
it('should complete within time limit', async () => {
  const promise = system.longRunningOperation();
  await expect(promise).resolves.toBeCompleted({ timeout: 5000 });
});

// 並行処理のテスト
it('should handle concurrent requests', async () => {
  const promises = Array(10).fill(null).map(() => 
    system.processInput(generateInput())
  );
  
  const results = await Promise.all(promises);
  expect(results).toHaveLength(10);
  expect(results.every(r => r.isValid)).toBe(true);
});
```

### 5. エラーケースのテスト

```typescript
describe('Error Handling', () => {
  it('should handle LLM timeout gracefully', async () => {
    const mockLLM = new MockLLMGateway();
    mockLLM.simulateTimeout();
    
    const system = new ConsciousnessSystem({ llm: mockLLM });
    
    await expect(system.generateResponse("test"))
      .rejects
      .toThrow(LLMTimeoutError);
    
    // システムは回復可能な状態を維持
    expect(system.isOperational()).toBe(true);
  });
  
  it('should fallback when consciousness level is invalid', () => {
    const invalidConsciousness = new ConsciousnessState({ phi: -1 });
    
    expect(() => system.process(input, invalidConsciousness))
      .toThrow(InvalidConsciousnessError);
  });
});
```

---

## まとめ

TDDは人工意識システムの開発において以下の価値を提供します：

1. **複雑性の管理**: 小さなステップで着実に前進
2. **品質の保証**: 継続的な検証による高品質な実装
3. **設計の改善**: テストファーストによる使いやすいAPI
4. **ドキュメント化**: テストが生きた仕様書として機能
5. **信頼性**: 非決定的な振る舞いも適切にテスト

特にLLM統合においては、プロパティベーステスト、モックサービス、契約テストなどの手法を組み合わせることで、非決定的な性質を持つシステムでも信頼性の高いテストが可能です。

「テストがなければ、それは動かない」 - この原則を守ることで、意識という複雑な現象の実装においても、着実な進歩が可能となります。