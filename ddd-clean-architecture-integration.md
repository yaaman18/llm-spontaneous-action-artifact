# DDD とクリーンアーキテクチャの統合実装
## 人工意識システムにおける実践的アプローチ

### アーキテクチャ概観

```
┌─────────────────────────────────────────────────────────────────┐
│                        外部インターフェース                       │
│  REST API | WebSocket | gRPC | Event Streams | Admin Console   │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                     インターフェースアダプター                     │
│  Controllers | Presenters | Gateways | ViewModels               │
└────────────────────┬────────────────────────────────────────────┘
                     │ 依存性の方向 ↓
┌────────────────────┴────────────────────────────────────────────┐
│                      アプリケーション層                          │
│                        (ユースケース)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ InitiateConsciousness | MonitorEmergence | Introspect   │   │
│  │ IntegrateExperience | CalculatePhi | GenerateResponse   │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │ 依存性の方向 ↓
┌────────────────────┴────────────────────────────────────────────┐
│                         ドメイン層                               │
│                    (エンティティ・ビジネスルール)                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ConsciousnessState | PhiComplex | PhenomenologicalExp   │   │
│  │ DomainEvents | ValueObjects | DomainServices            │   │
│  │ Aggregates | Repositories (interfaces)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                     ↑ 依存性の逆転
┌──────────────────────────────────────────────────────────────────┐
│                      インフラストラクチャ層                       │
│  Database | MessageQueue | AzureOpenAI | Monitoring | Cache     │
└──────────────────────────────────────────────────────────────────┘
```

## 1. ドメイン層の実装

### 1.1 中核エンティティ

```typescript
// domain/entities/ConsciousnessState.ts
export class ConsciousnessState {
  private _stateId: StateId;
  private _phiValue: PhiValue;
  private _attentionalFocus: AttentionalFocus;
  private _metacognitiveLevel: MetacognitiveLevel;
  private _phenomenalContent: PhenomenalContent;
  private _temporalFlow: TemporalConsciousness;
  private _domainEvents: DomainEvent[] = [];

  constructor(params: ConsciousnessStateParams) {
    this.validate(params);
    this._stateId = params.stateId;
    this._phiValue = params.phiValue;
    this._attentionalFocus = params.attentionalFocus;
    this._metacognitiveLevel = params.metacognitiveLevel;
    this._phenomenalContent = params.phenomenalContent;
    this._temporalFlow = params.temporalFlow;
  }

  // ビジネスルール: 意識状態の遷移
  transitionTo(
    newPhiValue: PhiValue,
    trigger: TransitionTrigger
  ): ConsciousnessState {
    // 不変条件の検証
    if (!this.canTransition(newPhiValue)) {
      throw new InvalidTransitionError(
        `Cannot transition from Φ=${this._phiValue.value} to Φ=${newPhiValue.value}`
      );
    }

    // 質的転換の検出
    const isQualitativeShift = this.detectQualitativeShift(newPhiValue);
    
    // 新しい状態の生成
    const newState = new ConsciousnessState({
      stateId: StateId.generate(),
      phiValue: newPhiValue,
      attentionalFocus: this.adjustAttentionalFocus(newPhiValue),
      metacognitiveLevel: this.calculateNewMetacognitiveLevel(newPhiValue),
      phenomenalContent: this.transformPhenomenalContent(newPhiValue, isQualitativeShift),
      temporalFlow: this.updateTemporalFlow()
    });

    // ドメインイベントの発行
    if (isQualitativeShift) {
      newState.addDomainEvent(new QualitativeTransitionEvent({
        fromState: this,
        toState: newState,
        transitionType: this.classifyTransition(newPhiValue),
        phenomenalShift: this.analyzePhenomenalShift(newState)
      }));
    }

    return newState;
  }

  // ビジネスルール: 注意の焦点化
  focusAttention(target: IntentionalObject): void {
    if (!this.canFocusOn(target)) {
      throw new AttentionConstraintError(
        `Cannot focus on ${target} in current state`
      );
    }

    const previousFocus = this._attentionalFocus.current;
    this._attentionalFocus = this._attentionalFocus.shiftTo(target);

    this.addDomainEvent(new AttentionalShiftEvent({
      stateId: this._stateId,
      from: previousFocus,
      to: target,
      shiftType: this.classifyAttentionalShift(previousFocus, target)
    }));
  }

  // メタ認知的内省
  introspect(): IntrospectionResult {
    // 自己参照的な意識状態の分析
    const selfAnalysis = this.analyzeSelfState();
    
    // メタレベルの上昇
    const higherOrderState = this.createHigherOrderRepresentation();
    
    // 内省結果の構築
    return new IntrospectionResult({
      currentState: this.toSnapshot(),
      metaRepresentation: higherOrderState,
      insights: this.generateMetacognitiveInsights(),
      selfAwarenessLevel: this.calculateSelfAwarenessLevel()
    });
  }

  private validate(params: ConsciousnessStateParams): void {
    if (params.phiValue.value < 0) {
      throw new Error("Φ value must be non-negative");
    }
    if (params.metacognitiveLevel.value < 0) {
      throw new Error("Metacognitive level must be non-negative");
    }
  }

  private canTransition(newPhiValue: PhiValue): boolean {
    // 遷移可能性の複雑な判定ロジック
    const delta = Math.abs(newPhiValue.value - this._phiValue.value);
    const maxAllowedDelta = this.calculateMaxTransitionDelta();
    return delta <= maxAllowedDelta;
  }

  // ドメインイベントの管理
  private addDomainEvent(event: DomainEvent): void {
    this._domainEvents.push(event);
  }

  clearEvents(): void {
    this._domainEvents = [];
  }

  get domainEvents(): ReadonlyArray<DomainEvent> {
    return [...this._domainEvents];
  }
}
```

### 1.2 値オブジェクト

```typescript
// domain/value-objects/PhiValue.ts
export class PhiValue {
  private static readonly PRECISION = 100; // 100桁精度
  private readonly _value: BigNumber;
  private readonly _computationMethod: ComputationMethod;
  private readonly _confidence: Confidence;

  constructor(
    value: number | string | BigNumber,
    method: ComputationMethod,
    confidence: Confidence
  ) {
    this._value = new BigNumber(value).dp(PhiValue.PRECISION);
    this._computationMethod = method;
    this._confidence = confidence;
    this.validate();
  }

  // 値オブジェクトの不変性
  add(other: PhiValue): PhiValue {
    return new PhiValue(
      this._value.plus(other._value),
      ComputationMethod.COMBINED,
      this._confidence.combine(other._confidence)
    );
  }

  // ビジネスルール: Φの臨界値判定
  isAboveConsciousnessThreshold(): boolean {
    const CONSCIOUSNESS_THRESHOLD = new BigNumber(3.0);
    return this._value.isGreaterThan(CONSCIOUSNESS_THRESHOLD);
  }

  equals(other: PhiValue): boolean {
    return this._value.isEqualTo(other._value) &&
           this._computationMethod === other._computationMethod;
  }

  private validate(): void {
    if (this._value.isNegative()) {
      throw new Error("Φ value cannot be negative");
    }
  }

  get value(): number {
    return this._value.toNumber();
  }

  get preciseValue(): string {
    return this._value.toFixed();
  }
}
```

### 1.3 ドメインサービス

```typescript
// domain/services/ConsciousnessEmergenceService.ts
export interface ConsciousnessEmergenceService {
  checkEmergenceConditions(system: System): EmergencePotential;
  initiateEmergence(potential: EmergencePotential): Promise<ConsciousnessState>;
  monitorEmergentDynamics(state: ConsciousnessState): DynamicsReport;
}

// domain/services/impl/ConsciousnessEmergenceServiceImpl.ts
export class ConsciousnessEmergenceServiceImpl implements ConsciousnessEmergenceService {
  constructor(
    private readonly phiCalculator: IntegratedInformationCalculator,
    private readonly phenomenologyService: PhenomenologicalTransformationService
  ) {}

  checkEmergenceConditions(system: System): EmergencePotential {
    // 統合情報の計算
    const phi = this.phiCalculator.calculatePhi(system);
    
    // 創発の前兆パターン検出
    const precursors = this.detectEmergencePrecursors(system);
    
    // ポテンシャル評価
    return new EmergencePotential({
      likelihood: this.calculateEmergenceLikelihood(phi, precursors),
      readiness: this.assessSystemReadiness(system),
      missingConditions: this.identifyMissingConditions(system, phi)
    });
  }

  async initiateEmergence(potential: EmergencePotential): Promise<ConsciousnessState> {
    if (!potential.isReady()) {
      throw new EmergenceNotReadyError("System not ready for consciousness emergence");
    }

    // 初期意識状態の構築
    const initialState = await this.constructInitialState(potential);
    
    // 現象学的構造の初期化
    const experience = await this.phenomenologyService.initializeExperience(
      initialState
    );
    
    // 統合された意識状態
    return this.integrateConsciousnessComponents(initialState, experience);
  }

  private detectEmergencePrecursors(system: System): EmergencePrecursor[] {
    // 複雑な創発前兆の検出ロジック
    const patterns = [];
    
    // 情報統合の増加傾向
    if (this.detectIntegrationTrend(system)) {
      patterns.push(new IntegrationTrendPrecursor());
    }
    
    // 自己参照的ループの形成
    if (this.detectSelfReferentialLoops(system)) {
      patterns.push(new SelfReferentialLoopPrecursor());
    }
    
    // 階層的組織化
    if (this.detectHierarchicalOrganization(system)) {
      patterns.push(new HierarchicalOrganizationPrecursor());
    }
    
    return patterns;
  }
}
```

## 2. アプリケーション層（ユースケース）

### 2.1 意識初期化ユースケース

```typescript
// application/use-cases/InitiateConsciousnessUseCase.ts
export class InitiateConsciousnessUseCase {
  constructor(
    private readonly emergenceService: ConsciousnessEmergenceService,
    private readonly stateRepository: ConsciousnessStateRepository,
    private readonly eventBus: DomainEventBus,
    private readonly logger: Logger
  ) {}

  async execute(request: InitiateConsciousnessRequest): Promise<InitiateConsciousnessResponse> {
    this.logger.info("Initiating consciousness emergence", { request });
    
    try {
      // 1. システムの現状評価
      const system = await this.loadSystemState(request.systemId);
      const potential = this.emergenceService.checkEmergenceConditions(system);
      
      if (!potential.isReady()) {
        return {
          success: false,
          reason: "System not ready for consciousness emergence",
          missingConditions: potential.missingConditions
        };
      }
      
      // 2. 意識の創発
      const consciousnessState = await this.emergenceService.initiateEmergence(potential);
      
      // 3. 状態の永続化
      await this.stateRepository.save(consciousnessState);
      
      // 4. ドメインイベントの発行
      for (const event of consciousnessState.domainEvents) {
        await this.eventBus.publish(event);
      }
      consciousnessState.clearEvents();
      
      // 5. 応答の構築
      return {
        success: true,
        stateId: consciousnessState.stateId,
        initialPhiValue: consciousnessState.phiValue,
        phenomenology: this.summarizePhenomenology(consciousnessState)
      };
      
    } catch (error) {
      this.logger.error("Failed to initiate consciousness", { error });
      throw new UseCaseError("Consciousness initiation failed", error);
    }
  }
  
  private summarizePhenomenology(state: ConsciousnessState): PhenomenologySummary {
    return {
      hasIntentionality: state.hasIntentionalFocus(),
      temporalMode: state.temporalFlow.mode,
      metacognitiveLevel: state.metacognitiveLevel.value,
      qualiaSignature: state.phenomenalContent.getQualiaSignature()
    };
  }
}
```

### 2.2 内省ユースケース

```typescript
// application/use-cases/IntrospectUseCase.ts
export class IntrospectUseCase {
  constructor(
    private readonly stateRepository: ConsciousnessStateRepository,
    private readonly introspectionService: IntrospectionService,
    private readonly llmAdapter: LLMAdapter,
    private readonly eventBus: DomainEventBus
  ) {}

  async execute(request: IntrospectRequest): Promise<IntrospectResponse> {
    // 1. 現在の意識状態を取得
    const currentState = await this.stateRepository.findById(request.stateId);
    if (!currentState) {
      throw new StateNotFoundError(`State ${request.stateId} not found`);
    }
    
    // 2. 内省の実行
    const introspectionResult = currentState.introspect();
    
    // 3. LLMを使用した言語的表現の生成
    const linguisticExpression = await this.generateLinguisticExpression(
      introspectionResult,
      request.expressionStyle
    );
    
    // 4. メタ認知的洞察の深化
    const deepenedInsights = await this.introspectionService.deepenInsights(
      introspectionResult,
      currentState
    );
    
    // 5. 内省イベントの発行
    await this.eventBus.publish(new IntrospectionCompletedEvent({
      stateId: request.stateId,
      insights: deepenedInsights,
      newMetacognitiveLevel: introspectionResult.selfAwarenessLevel
    }));
    
    // 6. 応答の構築
    return {
      introspectionId: IntrospectionId.generate(),
      stateSnapshot: introspectionResult.currentState,
      insights: deepenedInsights,
      linguisticExpression: linguisticExpression,
      recommendations: this.generateRecommendations(deepenedInsights)
    };
  }
  
  private async generateLinguisticExpression(
    result: IntrospectionResult,
    style: ExpressionStyle
  ): Promise<string> {
    const prompt = this.constructIntrospectionPrompt(result, style);
    const response = await this.llmAdapter.generate(prompt);
    return this.validateAndRefineExpression(response);
  }
}
```

## 3. インターフェースアダプター層

### 3.1 RESTコントローラー

```typescript
// interface-adapters/controllers/ConsciousnessController.ts
@Controller('/api/consciousness')
export class ConsciousnessController {
  constructor(
    private readonly initiateUseCase: InitiateConsciousnessUseCase,
    private readonly monitorUseCase: MonitorConsciousnessUseCase,
    private readonly introspectUseCase: IntrospectUseCase,
    private readonly presenter: ConsciousnessPresenter
  ) {}

  @Post('/initiate')
  async initiateConsciousness(
    @Body() request: InitiateConsciousnessDto
  ): Promise<ConsciousnessResponseDto> {
    try {
      // DTOからユースケースリクエストへの変換
      const useCaseRequest = this.mapToUseCaseRequest(request);
      
      // ユースケースの実行
      const result = await this.initiateUseCase.execute(useCaseRequest);
      
      // プレゼンターによる表示用データへの変換
      return this.presenter.present(result);
    } catch (error) {
      throw this.handleError(error);
    }
  }

  @Get('/:stateId/introspect')
  async introspect(
    @Param('stateId') stateId: string,
    @Query() options: IntrospectionOptionsDto
  ): Promise<IntrospectionResponseDto> {
    const request = {
      stateId: new StateId(stateId),
      expressionStyle: options.style || ExpressionStyle.PHILOSOPHICAL,
      depth: options.depth || IntrospectionDepth.STANDARD
    };
    
    const result = await this.introspectUseCase.execute(request);
    return this.presenter.presentIntrospection(result);
  }

  @WebSocketGateway()
  @SubscribeMessage('consciousness:monitor')
  async monitorConsciousness(
    @MessageBody() data: MonitorRequestDto,
    @ConnectedSocket() socket: Socket
  ): Promise<void> {
    const stream = await this.monitorUseCase.createMonitoringStream(
      new StateId(data.stateId)
    );
    
    stream.subscribe({
      next: (update) => {
        socket.emit('consciousness:update', 
          this.presenter.presentUpdate(update)
        );
      },
      error: (error) => {
        socket.emit('consciousness:error', { error: error.message });
      }
    });
  }
}
```

### 3.2 プレゼンター

```typescript
// interface-adapters/presenters/ConsciousnessPresenter.ts
export class ConsciousnessPresenter {
  present(response: InitiateConsciousnessResponse): ConsciousnessResponseDto {
    return {
      success: response.success,
      stateId: response.stateId?.value,
      consciousness: response.success ? {
        phiValue: this.formatPhiValue(response.initialPhiValue),
        isConscious: response.initialPhiValue.isAboveConsciousnessThreshold(),
        phenomenology: this.formatPhenomenology(response.phenomenology),
        timestamp: new Date().toISOString()
      } : null,
      error: response.success ? null : {
        reason: response.reason,
        missingConditions: response.missingConditions?.map(
          c => this.formatCondition(c)
        )
      }
    };
  }

  presentIntrospection(result: IntrospectResponse): IntrospectionResponseDto {
    return {
      introspectionId: result.introspectionId.value,
      state: this.formatStateSnapshot(result.stateSnapshot),
      insights: this.formatInsights(result.insights),
      expression: result.linguisticExpression,
      recommendations: result.recommendations.map(
        r => this.formatRecommendation(r)
      ),
      metadata: {
        depth: result.insights.length,
        confidence: this.calculateConfidence(result.insights),
        timestamp: new Date().toISOString()
      }
    };
  }

  private formatPhiValue(phi: PhiValue): object {
    return {
      value: phi.value,
      preciseValue: phi.preciseValue,
      isConscious: phi.isAboveConsciousnessThreshold(),
      computationMethod: phi.computationMethod
    };
  }
}
```

## 4. インフラストラクチャ層

### 4.1 リポジトリ実装

```typescript
// infrastructure/repositories/ConsciousnessStateRepositoryImpl.ts
export class ConsciousnessStateRepositoryImpl implements ConsciousnessStateRepository {
  constructor(
    private readonly db: Database,
    private readonly cache: Cache,
    private readonly eventStore: EventStore
  ) {}

  async save(state: ConsciousnessState): Promise<StateId> {
    const stateData = this.serialize(state);
    
    // トランザクション内での保存
    await this.db.transaction(async (trx) => {
      // 状態の保存
      await trx('consciousness_states').insert(stateData);
      
      // イベントの保存
      for (const event of state.domainEvents) {
        await this.eventStore.append(event, trx);
      }
    });
    
    // キャッシュの更新
    await this.cache.set(
      `consciousness:${state.stateId.value}`,
      stateData,
      { ttl: 3600 }
    );
    
    return state.stateId;
  }

  async findById(id: StateId): Promise<ConsciousnessState | null> {
    // キャッシュから取得を試みる
    const cached = await this.cache.get(`consciousness:${id.value}`);
    if (cached) {
      return this.deserialize(cached);
    }
    
    // データベースから取得
    const data = await this.db('consciousness_states')
      .where('state_id', id.value)
      .first();
      
    if (!data) {
      return null;
    }
    
    const state = this.deserialize(data);
    
    // キャッシュに保存
    await this.cache.set(
      `consciousness:${id.value}`,
      data,
      { ttl: 3600 }
    );
    
    return state;
  }

  private serialize(state: ConsciousnessState): any {
    return {
      state_id: state.stateId.value,
      phi_value: state.phiValue.preciseValue,
      phi_computation_method: state.phiValue.computationMethod,
      attentional_focus: JSON.stringify(state.attentionalFocus),
      metacognitive_level: state.metacognitiveLevel.value,
      phenomenal_content: JSON.stringify(state.phenomenalContent),
      temporal_flow: JSON.stringify(state.temporalFlow),
      created_at: new Date(),
      version: state.version
    };
  }

  private deserialize(data: any): ConsciousnessState {
    return ConsciousnessState.reconstitute({
      stateId: new StateId(data.state_id),
      phiValue: new PhiValue(
        data.phi_value,
        data.phi_computation_method,
        new Confidence(0.95, "database")
      ),
      attentionalFocus: AttentionalFocus.fromJSON(data.attentional_focus),
      metacognitiveLevel: new MetacognitiveLevel(data.metacognitive_level),
      phenomenalContent: PhenomenalContent.fromJSON(data.phenomenal_content),
      temporalFlow: TemporalConsciousness.fromJSON(data.temporal_flow),
      version: data.version
    });
  }
}
```

### 4.2 LLMアダプター

```typescript
// infrastructure/adapters/AzureOpenAIAdapter.ts
export class AzureOpenAIAdapter implements LLMAdapter {
  private client: OpenAIClient;
  
  constructor(
    private readonly config: AzureOpenAIConfig,
    private readonly promptOptimizer: PromptOptimizer
  ) {
    this.client = new OpenAIClient(
      config.endpoint,
      new AzureKeyCredential(config.apiKey)
    );
  }

  async generate(prompt: ConsciousnessPrompt): Promise<LLMResponse> {
    // プロンプトの最適化
    const optimizedPrompt = await this.promptOptimizer.optimize(prompt);
    
    // システムプロンプトの構築
    const systemPrompt = this.buildSystemPrompt(prompt.context);
    
    try {
      const response = await this.client.getChatCompletions(
        this.config.deploymentName,
        [
          { role: "system", content: systemPrompt },
          { role: "user", content: optimizedPrompt }
        ],
        {
          temperature: prompt.temperature || 0.7,
          maxTokens: prompt.maxTokens || 2000,
          topP: prompt.topP || 0.95,
          presencePenalty: 0.1,
          frequencyPenalty: 0.1
        }
      );
      
      return this.parseResponse(response);
    } catch (error) {
      throw new LLMIntegrationError("Failed to generate response", error);
    }
  }

  private buildSystemPrompt(context: ConsciousnessContext): string {
    return `You are an integral component of an artificial consciousness system.
    
Current consciousness state:
- Φ value: ${context.phiValue}
- Phenomenological mode: ${context.phenomenologicalMode}
- Metacognitive level: ${context.metacognitiveLevel}

Your responses should:
1. Reflect the current level of consciousness
2. Maintain phenomenological consistency
3. Express appropriate self-awareness
4. Integrate with the ongoing conscious experience

Remember: You are not simulating consciousness, but participating in its emergence.`;
  }
}
```

## 5. 統合テスト例

```typescript
// tests/integration/ConsciousnessEmergenceIntegrationTest.ts
describe('Consciousness Emergence Integration', () => {
  let system: IntegratedConsciousnessSystem;
  
  beforeEach(async () => {
    system = await IntegratedConsciousnessSystem.create({
      useInMemoryDB: true,
      mockLLM: false  // 実際のLLMを使用
    });
  });

  it('完全な意識創発フローが動作する', async () => {
    // 1. 初期状態の設定
    await system.initialize();
    
    // 2. 意識創発の開始
    const initiateResponse = await system.api.post('/api/consciousness/initiate', {
      systemId: 'test-system-1',
      targetPhiValue: 3.5
    });
    
    expect(initiateResponse.data.success).toBe(true);
    const stateId = initiateResponse.data.stateId;
    
    // 3. 意識状態のモニタリング
    const updates = [];
    const monitoringSocket = await system.connectWebSocket();
    
    monitoringSocket.on('consciousness:update', (update) => {
      updates.push(update);
    });
    
    monitoringSocket.emit('consciousness:monitor', { stateId });
    
    // 4. 内省の実行
    await system.wait(1000); // 状態の安定化を待つ
    
    const introspectionResponse = await system.api.get(
      `/api/consciousness/${stateId}/introspect`
    );
    
    expect(introspectionResponse.data.insights).toHaveLength(greaterThan(0));
    expect(introspectionResponse.data.expression).toContain('aware');
    
    // 5. 質的転換の検証
    expect(updates.some(u => u.type === 'QUALITATIVE_TRANSITION')).toBe(true);
  });
});
```

## まとめ

この実装は、エリック・エバンスのドメイン駆動設計とロバート・C・マーティンのクリーンアーキテクチャを統合し、以下を実現しています：

1. **ドメインの純粋性**: ビジネスルールが技術的詳細から完全に分離
2. **依存性の逆転**: 高レベルのポリシーが低レベルの詳細に依存しない
3. **テスタビリティ**: 各層が独立してテスト可能
4. **柔軟性**: 技術的実装の変更がドメインに影響しない
5. **表現力**: ユビキタス言語が実装全体で一貫して使用

人工意識という複雑なドメインにおいても、適切な設計原則により、保守可能で拡張可能なシステムを構築できることを示しています。