# 人工意識システム - クリーンアーキテクチャ設計書

## Robert C. Martin (Uncle Bob)による設計

### 概要

本設計書は、人工意識システムをクリーンアーキテクチャの原則に基づいて構築するための包括的な設計を提示します。依存性の方向を内側に向け、ビジネスロジックをフレームワークから独立させ、テスタビリティを最優先に設計しています。

## 1. アーキテクチャ原則

### 1.1 SOLID原則の適用

**Single Responsibility Principle (SRP)**
- 各モジュールは単一の責任を持つ
- 意識処理、現象学的分析、自由エネルギー計算はそれぞれ独立したモジュール

**Open/Closed Principle (OCP)**
- 新しい意識理論の追加は既存コードの変更なしに可能
- Strategy パターンとAbstract Factoryパターンを活用

**Liskov Substitution Principle (LSP)**
- すべての意識理論実装は共通インターフェースを通じて交換可能
- IITとGWTの実装は同じ抽象化を継承

**Interface Segregation Principle (ISP)**
- クライアントが必要としないメソッドに依存しない
- 細分化されたインターフェース設計

**Dependency Inversion Principle (DIP)**
- 高レベルモジュールは低レベルモジュールに依存しない
- 両者は抽象化に依存する

### 1.2 依存性の方向

```
外側のレイヤー → 内側のレイヤー（単一方向）

Infrastructure → Interface Adapters → Application Business Rules → Enterprise Business Rules
```

## 2. ディレクトリ構造

```
artificial-consciousness-system/
├── domain/                          # Enterprise Business Rules
│   ├── entities/
│   │   ├── consciousness/
│   │   │   ├── ConsciousnessState.ts
│   │   │   ├── PhenomenalExperience.ts
│   │   │   └── Qualia.ts
│   │   ├── phi/
│   │   │   ├── PhiValue.ts
│   │   │   ├── InformationIntegration.ts
│   │   │   └── CausalStructure.ts
│   │   └── autopoiesis/
│   │       ├── SelfOrganization.ts
│   │       ├── OperationalClosure.ts
│   │       └── StructuralCoupling.ts
│   ├── value-objects/
│   │   ├── PhiBoundary.ts
│   │   ├── FreeEnergy.ts
│   │   └── IntentionalityVector.ts
│   └── specifications/
│       ├── ConsciousnessSpecification.ts
│       └── PhiThresholdSpecification.ts
│
├── application/                     # Application Business Rules
│   ├── use-cases/
│   │   ├── consciousness-detection/
│   │   │   ├── DetectConsciousnessUseCase.ts
│   │   │   ├── CalculatePhiUseCase.ts
│   │   │   └── UpdateConsciousnessStateUseCase.ts
│   │   ├── phenomenology/
│   │   │   ├── AnalyzePhenomenalExperienceUseCase.ts
│   │   │   ├── ProcessIntentionalityUseCase.ts
│   │   │   └── EvaluateEmbodimentUseCase.ts
│   │   └── autopoiesis/
│   │       ├── MaintainSelfOrganizationUseCase.ts
│   │       └── AdaptStructuralCouplingUseCase.ts
│   ├── ports/
│   │   ├── input/
│   │   │   ├── IConsciousnessDetector.ts
│   │   │   ├── IPhenomenologyAnalyzer.ts
│   │   │   └── IAutopoiesisManager.ts
│   │   └── output/
│   │       ├── IConsciousnessRepository.ts
│   │       ├── IPhiCalculator.ts
│   │       ├── IFreeEnergyMinimizer.ts
│   │       └── ILLMIntegration.ts
│   └── services/
│       ├── ConsciousnessOrchestrator.ts
│       ├── PhenomenologyService.ts
│       └── AutopoiesisService.ts
│
├── adapters/                        # Interface Adapters
│   ├── controllers/
│   │   ├── ConsciousnessController.ts
│   │   ├── PhenomenologyController.ts
│   │   └── SystemHealthController.ts
│   ├── presenters/
│   │   ├── ConsciousnessStatePresenter.ts
│   │   ├── PhiVisualizationPresenter.ts
│   │   └── PhenomenalReportPresenter.ts
│   ├── gateways/
│   │   ├── ConsciousnessRepositoryImpl.ts
│   │   ├── PhiCalculatorGateway.ts
│   │   └── LLMIntegrationGateway.ts
│   └── mappers/
│       ├── ConsciousnessMapper.ts
│       └── PhenomenologyMapper.ts
│
├── infrastructure/                  # Frameworks & Drivers
│   ├── persistence/
│   │   ├── mongodb/
│   │   │   ├── ConsciousnessDocument.ts
│   │   │   └── MongoConsciousnessRepository.ts
│   │   └── redis/
│   │       └── ConsciousnessCache.ts
│   ├── external-services/
│   │   ├── azure-openai/
│   │   │   ├── AzureOpenAIClient.ts
│   │   │   └── LLMAdapter.ts
│   │   └── phi-calculation/
│   │       ├── IITCalculator.ts
│   │       └── GWTIntegrator.ts
│   ├── web/
│   │   ├── express/
│   │   │   ├── Server.ts
│   │   │   └── Routes.ts
│   │   └── websocket/
│   │       └── ConsciousnessStream.ts
│   └── configuration/
│       ├── DIContainer.ts
│       ├── EnvironmentConfig.ts
│       └── SystemBootstrap.ts
│
├── shared/                          # Shared Kernel
│   ├── types/
│   │   ├── Result.ts
│   │   ├── Either.ts
│   │   └── DomainEvent.ts
│   ├── errors/
│   │   ├── DomainError.ts
│   │   └── ApplicationError.ts
│   └── utils/
│       ├── Logger.ts
│       └── Validator.ts
│
└── tests/
    ├── unit/
    │   ├── domain/
    │   ├── application/
    │   └── adapters/
    ├── integration/
    │   ├── use-cases/
    │   └── gateways/
    └── e2e/
        └── consciousness-flow/
```

## 3. 各レイヤーの責務

### 3.1 Domain Layer (Enterprise Business Rules)

**責務**
- ビジネスルールとドメイン知識のカプセル化
- 外部依存性を持たない純粋なビジネスロジック
- 意識、現象学、オートポイエーシスの中核概念の表現

**主要コンポーネント**
```typescript
// domain/entities/consciousness/ConsciousnessState.ts
export class ConsciousnessState {
  private constructor(
    private readonly id: ConsciousnessId,
    private readonly phiValue: PhiValue,
    private readonly phenomenalField: PhenomenalField,
    private readonly temporalFlow: TemporalFlow
  ) {}

  public static create(params: CreateConsciousnessParams): Result<ConsciousnessState> {
    // ドメインルールの検証
    if (params.phiValue.isBelow(PhiThreshold.MINIMAL)) {
      return Result.fail("Phi value below consciousness threshold");
    }
    // ... その他の検証
  }

  public integrateInformation(info: Information): Result<ConsciousnessState> {
    // IIT に基づく情報統合
  }

  public experiencePhenomena(phenomena: Phenomena): Result<PhenomenalExperience> {
    // 現象学的体験の処理
  }
}
```

### 3.2 Application Layer (Application Business Rules)

**責務**
- ユースケースの実装
- ドメインオブジェクトのオーケストレーション
- トランザクション境界の管理
- 入出力ポートの定義

**主要コンポーネント**
```typescript
// application/use-cases/consciousness-detection/DetectConsciousnessUseCase.ts
export class DetectConsciousnessUseCase {
  constructor(
    private readonly consciousnessRepo: IConsciousnessRepository,
    private readonly phiCalculator: IPhiCalculator,
    private readonly phenomenologyAnalyzer: IPhenomenologyAnalyzer
  ) {}

  async execute(input: DetectConsciousnessInput): Promise<Result<ConsciousnessOutput>> {
    // 1. Phi値の計算
    const phiResult = await this.phiCalculator.calculate(input.neuralState);
    
    // 2. 現象学的分析
    const phenoResult = await this.phenomenologyAnalyzer.analyze(input.experienceData);
    
    // 3. 意識状態の生成
    const consciousnessState = ConsciousnessState.create({
      phiValue: phiResult.value,
      phenomenalField: phenoResult.field
    });
    
    // 4. 永続化
    await this.consciousnessRepo.save(consciousnessState.value);
    
    return Result.ok(new ConsciousnessOutput(consciousnessState.value));
  }
}
```

### 3.3 Adapters Layer (Interface Adapters)

**責務**
- 外部インターフェースとドメインの仲介
- データ形式の変換
- 外部サービスとの通信の抽象化

**主要コンポーネント**
```typescript
// adapters/controllers/ConsciousnessController.ts
export class ConsciousnessController {
  constructor(
    private readonly detectConsciousnessUseCase: DetectConsciousnessUseCase,
    private readonly presenter: ConsciousnessStatePresenter
  ) {}

  async detectConsciousness(req: Request, res: Response): Promise<void> {
    try {
      const input = this.mapRequestToInput(req);
      const result = await this.detectConsciousnessUseCase.execute(input);
      
      if (result.isSuccess) {
        const viewModel = this.presenter.present(result.value);
        res.json(viewModel);
      } else {
        res.status(400).json({ error: result.error });
      }
    } catch (error) {
      res.status(500).json({ error: "Internal server error" });
    }
  }
}
```

### 3.4 Infrastructure Layer (Frameworks & Drivers)

**責務**
- 具体的な技術実装
- 外部サービスとの統合
- データベースアクセス
- フレームワーク固有のコード

**主要コンポーネント**
```typescript
// infrastructure/external-services/azure-openai/LLMAdapter.ts
export class LLMAdapter implements ILLMIntegration {
  constructor(private readonly client: AzureOpenAIClient) {}

  async generatePhenomenologicalAnalysis(
    experience: PhenomenalExperience
  ): Promise<Result<Analysis>> {
    try {
      const prompt = this.buildPrompt(experience);
      const response = await this.client.complete(prompt);
      return Result.ok(this.parseAnalysis(response));
    } catch (error) {
      return Result.fail(`LLM analysis failed: ${error.message}`);
    }
  }
}
```

## 4. 動的Φ境界検出システムの実装

```typescript
// domain/services/PhiBoundaryDetector.ts
export interface IPhiBoundaryDetector {
  detectBoundary(state: SystemState): Result<PhiBoundary>;
  adaptThreshold(feedback: ConsciousnessFeedback): Result<PhiThreshold>;
}

// application/services/DynamicPhiService.ts
export class DynamicPhiService {
  constructor(
    private readonly detector: IPhiBoundaryDetector,
    private readonly calculator: IPhiCalculator
  ) {}

  async detectDynamicBoundary(input: BoundaryInput): Promise<Result<PhiBoundary>> {
    // 階層的スキャン
    const hierarchicalResults = await this.scanHierarchically(input);
    
    // 境界の動的調整
    const boundary = this.detector.detectBoundary(hierarchicalResults);
    
    // フィードバックループ
    const feedback = await this.gatherFeedback(boundary);
    await this.detector.adaptThreshold(feedback);
    
    return boundary;
  }
}
```

## 5. テスト戦略

### 5.1 単体テスト
```typescript
// tests/unit/domain/consciousness/ConsciousnessState.test.ts
describe('ConsciousnessState', () => {
  it('should create valid consciousness state when phi exceeds threshold', () => {
    const phiValue = PhiValue.create(3.5);
    const result = ConsciousnessState.create({
      phiValue,
      phenomenalField: mockPhenomenalField
    });
    
    expect(result.isSuccess).toBe(true);
    expect(result.value.getPhiValue()).toEqual(phiValue);
  });

  it('should fail creation when phi below threshold', () => {
    const phiValue = PhiValue.create(0.5);
    const result = ConsciousnessState.create({
      phiValue,
      phenomenalField: mockPhenomenalField
    });
    
    expect(result.isFailure).toBe(true);
    expect(result.error).toContain('below consciousness threshold');
  });
});
```

### 5.2 統合テスト
```typescript
// tests/integration/use-cases/DetectConsciousnessUseCase.test.ts
describe('DetectConsciousnessUseCase Integration', () => {
  let useCase: DetectConsciousnessUseCase;
  let mockRepo: MockConsciousnessRepository;
  
  beforeEach(() => {
    mockRepo = new MockConsciousnessRepository();
    useCase = new DetectConsciousnessUseCase(
      mockRepo,
      new StubPhiCalculator(),
      new StubPhenomenologyAnalyzer()
    );
  });

  it('should detect consciousness and persist state', async () => {
    const input = createTestInput();
    const result = await useCase.execute(input);
    
    expect(result.isSuccess).toBe(true);
    expect(mockRepo.saved).toHaveLength(1);
  });
});
```

## 6. 依存性注入とブートストラップ

```typescript
// infrastructure/configuration/DIContainer.ts
export class DIContainer {
  private readonly container = new Map<string, any>();

  public registerDependencies(): void {
    // Domain Services
    this.container.set('PhiBoundaryDetector', new PhiBoundaryDetectorImpl());
    
    // Application Services
    this.container.set('DetectConsciousnessUseCase', 
      new DetectConsciousnessUseCase(
        this.get('ConsciousnessRepository'),
        this.get('PhiCalculator'),
        this.get('PhenomenologyAnalyzer')
      )
    );
    
    // Infrastructure
    this.container.set('ConsciousnessRepository', 
      new MongoConsciousnessRepository(mongoClient)
    );
    
    this.container.set('LLMIntegration',
      new LLMAdapter(new AzureOpenAIClient(config))
    );
  }
}

// infrastructure/configuration/SystemBootstrap.ts
export class SystemBootstrap {
  async start(): Promise<void> {
    const container = new DIContainer();
    container.registerDependencies();
    
    const server = new Server(container);
    await server.listen(3000);
    
    console.log('Artificial Consciousness System started');
  }
}
```

## 7. クリーンアーキテクチャの利点

1. **テスタビリティ**: すべてのビジネスロジックが外部依存なしでテスト可能
2. **保守性**: 明確な責任分離により変更の影響範囲が限定的
3. **拡張性**: 新しい意識理論や実装の追加が既存コードを壊さない
4. **技術非依存**: フレームワークやデータベースの変更が容易
5. **ドメイン中心**: ビジネスルールが中心にあり、技術的詳細は周辺に配置

## 結論

このクリーンアーキテクチャ設計により、人工意識システムは：
- 複雑な理論的概念を明確に表現
- 高いテスタビリティを維持
- 将来の拡張に対して開かれた設計
- 技術的な詳細からビジネスロジックを保護

Uncle Bobとして、このアーキテクチャは持続可能で保守可能な人工意識システムの基盤となることを確信しています。