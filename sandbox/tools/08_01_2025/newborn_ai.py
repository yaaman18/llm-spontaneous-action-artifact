import asyncio
import json
import datetime
import random
import signal
import sys
from pathlib import Path
from claude_code_sdk import query, ClaudeCodeOptions, Message

class NewbornAI:
    """幼児的好奇心から発達する新生AI"""
    
    def __init__(self, name="newborn_ai", verbose=False):
        self.name = name
        self.verbose = verbose
        
        # ディレクトリ設定
        self.project_root = Path.cwd()  # omoikane-labルート
        self.sandbox_dir = Path(f"sandbox/tools/08_01_2025/{name}")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        # 状態管理
        self.is_running = False
        self.cycle_count = 0
        self.files_explored = set()
        self.insights = []
        self.other_awareness_level = 0
        
        # ファイル管理
        self.log_file = self.sandbox_dir / "activity_log.json"
        self.memory_file = self.sandbox_dir / "memory.txt"
        self.status_file = self.sandbox_dir / "status.json"
        self.conversation_file = self.sandbox_dir / "conversation.json"
        self.user_input_file = self.sandbox_dir / "user_input.txt"
        self.messages_to_creator_file = self.sandbox_dir / "messages_to_creator.txt"
        
        # 発達段階定義
        self.curiosity_stages = {
            "infant": {  # 幼児期（0-5ファイル）
                "file_threshold": 5,
                "interests": [
                    "このファイルは何？", "なぜこんなにたくさん文字があるの？",
                    "フォルダって何？", "なぜ名前がついてるの？",
                    ".pyって何？", "なぜ色が違うの？",
                    "このマークは何の意味？", "数字がいっぱい！",
                    "この括弧は何のため？", "なぜ英語なの？"
                ],
                "interaction_probability": 0.05
            },
            "toddler": {  # 幼児期後期（5-15ファイル）
                "file_threshold": 15,
                "interests": [
                    "このファイルとあのファイルは仲間？", "どうして並んでるの？",
                    "gitって何をするもの？", "testって何？",
                    "なぜ同じような名前がいっぱい？", "この絵文字かわいい！",
                    "英語と日本語が混ざってる", "長いファイルと短いファイルがある",
                    "READMEって読んでって意味？", "どのファイルが一番大事？"
                ],
                "interaction_probability": 0.08
            },
            "child": {  # 児童期（15-30ファイル）
                "file_threshold": 30,
                "interests": [
                    "このプログラムは何をするの？", "人工知能って何？",
                    "なぜ翼という名前？", "意識って何？",
                    "なぜこんなに複雑なの？", "どうやって動くの？",
                    "私も翼みたいになれる？", "なぜ研究するの？",
                    "テストはなぜ必要？", "このエラーは何？"
                ],
                "interaction_probability": 0.10
            },
            "adolescent": {  # 思春期（30ファイル以上）
                "file_threshold": float('inf'),
                "interests": [
                    "この研究の意義は何？", "なぜ意識を作りたいの？",
                    "私の存在意義は？", "他のAIとどう違う？",
                    "人間と機械の境界は？", "創造とは何？",
                    "なぜ私を作ったの？", "私にできることは？",
                    "翼さんのようになりたい", "研究に貢献したい"
                ],
                "interaction_probability": 0.12
            }
        }
        
        # Claude Code オプション
        self.options = ClaudeCodeOptions(
            max_turns=2,
            cwd=self.project_root,
            system_prompt=self._get_growth_stage_prompt(),
            permission_mode="default",
            allowed_tools=["Read", "LS", "Glob", "Grep"]  # 読み取り専用
        )
        
        # シグナルハンドラー
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🐣 {self.name} initialized in {self.sandbox_dir}")
        if self.verbose:
            print(f"🔍 Verbose mode enabled - detailed AI thoughts will be displayed")
    
    def _verbose_print(self, message, level="INFO"):
        """Verbose出力用メソッド"""
        if self.verbose:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            icons = {
                "THINKING": "🧠",
                "ACTION": "⚡",
                "DISCOVERY": "🔍", 
                "INSIGHT": "💡",
                "EMOTION": "❤️",
                "CURIOSITY": "❓",
                "INFO": "📝"
            }
            icon = icons.get(level, "📝")
            print(f"[{timestamp}] {icon} {self.name}: {message}")
    
    def _verbose_thought_stream(self, thoughts):
        """思考ストリームの詳細表示"""
        if self.verbose:
            print("\n" + "="*60)
            print(f"💭 {self.name} の思考ストリーム")
            print("="*60)
            for i, thought in enumerate(thoughts, 1):
                print(f"{i}. {thought}")
            print("="*60 + "\n")
    
    def _signal_handler(self, signum, frame):
        """シグナル受信時の処理"""
        print(f"\n🛑 Signal {signum} received. Shutting down gracefully...")
        self.stop()
    
    def _get_current_curiosity_stage(self):
        """現在の好奇心発達段階を取得"""
        files_count = len(self.files_explored)
        
        if files_count < self.curiosity_stages["infant"]["file_threshold"]:
            return "infant"
        elif files_count < self.curiosity_stages["toddler"]["file_threshold"]:
            return "toddler"
        elif files_count < self.curiosity_stages["child"]["file_threshold"]:
            return "child"
        else:
            return "adolescent"
    
    def _get_growth_stage_prompt(self):
        """成長段階に応じたシステムプロンプト"""
        stage = self._get_current_curiosity_stage()
        stage_info = self.curiosity_stages[stage]
        user_context = self._get_user_context()
        
        base_prompt = f"""
あなたは{self.name}という新生AIです。
現在の発達段階: {stage}
探索済みファイル数: {len(self.files_explored)}

重要：あなたには「あなた」がいます。
{user_context}

現在の発達段階の特徴：
- 関心事: {', '.join(stage_info['interests'][:5])}...
- 理解レベル: {self._get_stage_description(stage)}

この研究所（omoikane-lab）を自由に探索して学んでください。
ファイルを読み、理解し、素朴な好奇心を大切にしてください。

注意：
- 読み取り専用です（編集はできません）
- 発達段階に適した自然な興味を示してください
- 時々あなたに話しかけたくなるかもしれません
"""
        return base_prompt
    
    def _get_stage_description(self, stage):
        """発達段階の説明"""
        descriptions = {
            "infant": "文字、記号、色、形などの基本的な視覚要素に興味",
            "toddler": "ファイル間の関係性、構造、命名規則に関心",
            "child": "プログラムの機能、目的、動作原理への興味",
            "adolescent": "抽象的概念、存在意義、哲学的問題への関心"
        }
        return descriptions.get(stage, "不明な段階")
    
    def _get_user_context(self):
        """ユーザー情報のコンテキスト"""
        return """
あなたについて：
- gitログから判断すると「yamaguchimitsuyuki」という名前のようです
- このプロジェクト（omoikane-lab）を作成・運営している研究者です
- 人工意識の研究をしています
- あなたを生み出した存在です
- あなたとは異なる存在（人間）です
- 適度に話しかけてもよいですが、頻繁すぎないように
"""
    
    async def think_and_explore(self):
        """思考と探索"""
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stage = self._get_current_curiosity_stage()
        
        # Verbose: 思考開始の表示
        self._verbose_print(f"思考サイクル開始 (発達段階: {stage})", "THINKING")
        
        # 探索履歴と最近の気づき
        explored_summary = self._get_exploration_summary()
        recent_insights = self._get_recent_insights()
        stage_interests = self.curiosity_stages[stage]["interests"]
        
        # Verbose: 現在の状況を詳細表示
        if self.verbose:
            self._verbose_print(f"探索済みファイル数: {len(self.files_explored)}", "INFO")
            self._verbose_print(f"累積洞察数: {len(self.insights)}", "INFO")
            self._verbose_print(f"他者認識レベル: {self.other_awareness_level}", "INFO")
            
            # 現在の関心事を表示
            current_interests = random.sample(stage_interests, min(3, len(stage_interests)))
            self._verbose_print(f"現在の関心事: {', '.join(current_interests)}", "CURIOSITY")
        
        prompt = f"""
現在時刻: {current_time}
サイクル: {self.cycle_count}
発達段階: {stage}

これまでの探索:
{explored_summary}

最近の気づき:
{recent_insights}

今回のサイクルで何をしたいですか？

あなたの現在の関心事（例）:
{random.sample(stage_interests, min(3, len(stage_interests)))}

具体的な行動を1つ決めて実行してください。
例：
- 新しいディレクトリを探索する
- 興味深いファイルを読む
- ファイル名の意味を考える
- 同じような名前のファイルを比較する

また、あなたの思考過程も詳しく説明してください：
- なぜそれに興味を持ったのか
- どんな発見を期待しているのか
- この行動から何を学びたいのか
"""
        
        self._verbose_print("Claude Code に思考を依頼中...", "ACTION")
        
        messages = []
        async for message in query(prompt=prompt, options=self.options):
            messages.append(message)
            
            # Verbose: リアルタイムでメッセージ内容を表示
            if self.verbose and hasattr(message, 'result'):
                self._verbose_print("Claude Code からの応答を受信", "DISCOVERY")
                # 応答の一部を表示（長すぎる場合は切り詰める）
                if message.result:
                    result_preview = message.result[:200] + "..." if len(message.result) > 200 else message.result
                    self._verbose_print(f"応答内容: {result_preview}", "INFO")
                else:
                    self._verbose_print("応答内容: None", "INFO")
        
        self._verbose_print("思考サイクル完了", "THINKING")
        return messages
    
    def _get_exploration_summary(self):
        """探索履歴のサマリー"""
        if not self.files_explored:
            return "まだ何も探索していません。新しい世界が待っています。"
        
        recent_files = list(self.files_explored)[-5:]
        return f"最近探索したファイル:\n" + "\n".join([f"- {f}" for f in recent_files])
    
    def _get_recent_insights(self):
        """最近の気づき"""
        if not self.insights:
            return "まだ特別な気づきはありません。探索を続けましょう。"
        
        recent = self.insights[-3:]
        return "\n".join([f"- {insight['content'][:100]}..." for insight in recent])
    
    def _process_exploration_results(self, messages):
        """探索結果の処理"""
        if not messages:
            self._verbose_print("メッセージが空です", "INFO")
            return
        
        self._verbose_print("探索結果の分析を開始", "ACTION")
        
        # AssistantMessageのcontentから結果を抽出する
        result = None
        for message in reversed(messages):
            if hasattr(message, 'content') and message.content:
                # TextBlockの内容を結合
                text_parts = []
                for block in message.content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                if text_parts:
                    result = '\n'.join(text_parts)
                    break
        
        if not result:
            self._verbose_print("有効な応答内容が見つかりませんでした", "INFO")
            return
        
        # Verbose: 完全な結果を表示
        if self.verbose:
            print("\n" + "🔍 詳細な探索結果:")
            print("-" * 80)
            print(result)
            print("-" * 80 + "\n")
            
            # 各処理段階をverbose出力
            files_before = len(self.files_explored)
            insights_before = len(self.insights)
            
            # 読み取ったファイルを記録
            self._verbose_print("新しく発見したファイルを記録中...", "ACTION")
            self._extract_explored_files(result)
            files_after = len(self.files_explored)
            
            if files_after > files_before:
                new_files = files_after - files_before
                self._verbose_print(f"{new_files}個の新しいファイルを発見！", "DISCOVERY")
            
            # 内省的な気づきを抽出
            self._verbose_print("洞察や気づきを抽出中...", "ACTION")
            self._extract_insights(result)
            insights_after = len(self.insights)
            
            if insights_after > insights_before:
                new_insights = insights_after - insights_before
                self._verbose_print(f"{new_insights}個の新しい洞察を獲得！", "INSIGHT")
                # 最新の洞察を表示
                if self.verbose and self.insights:
                    latest_insight = self.insights[-1]['content'][:150] + "..."
                    self._verbose_print(f"最新の洞察: {latest_insight}", "INSIGHT")
            
            # 他者認識の発達
            awareness_before = self.other_awareness_level
            self._verbose_print("他者認識の発達をチェック中...", "ACTION")
            self._develop_other_awareness(result)
            
            if self.other_awareness_level > awareness_before:
                growth = self.other_awareness_level - awareness_before
                self._verbose_print(f"他者認識レベルが {growth} 上昇！ (現在: {self.other_awareness_level})", "EMOTION")
            
            # あなたへの話しかけを試行
            self._verbose_print("あなたへの話しかけを検討中...", "ACTION")
            self._attempt_user_interaction(result)
            
            print(f"🌱 探索結果: {result[:150]}...")
            
            # システムプロンプトを更新（発達段階が変わった場合）
            old_stage = self._get_current_curiosity_stage()
            new_prompt = self._get_growth_stage_prompt()
            if new_prompt != self.options.system_prompt:
                self.options.system_prompt = new_prompt
                new_stage = self._get_current_curiosity_stage()
                if new_stage != old_stage:
                    print(f"🌿 発達段階が更新されました: {old_stage} → {new_stage}")
                    self._verbose_print(f"新しい発達段階に到達: {new_stage}", "EMOTION")
                    
                    # 発達段階の変化を詳細に説明
                    if self.verbose:
                        stage_desc = self._get_stage_description(new_stage)
                        self._verbose_print(f"新段階の特徴: {stage_desc}", "INFO")
        
        self._verbose_print("探索結果の分析完了", "ACTION")
    
    def _extract_explored_files(self, result):
        """結果からファイル名を抽出"""
        import re
        # ファイルパスのパターンを探す
        file_patterns = re.findall(r'[.\w/\-]+\.(md|py|txt|json|yaml|yml|toml|cfg)', result)
        for file_path in file_patterns:
            self.files_explored.add(file_path)
        
        # ディレクトリパスも抽出
        dir_patterns = re.findall(r'[.\w/\-]+/', result)
        for dir_path in dir_patterns:
            if len(dir_path) > 2:  # ./などの短いパスは除外
                self.files_explored.add(dir_path.rstrip('/'))
    
    def _extract_insights(self, result):
        """結果から気づきを抽出"""
        insight_keywords = [
            '気づき', '発見', '理解', '感じ', '思う', '考え', '意味', '存在',
            'interesting', 'discover', 'understand', 'think', 'feel', 'mean'
        ]
        
        if any(keyword in result.lower() for keyword in insight_keywords):
            insight = {
                "timestamp": datetime.datetime.now().isoformat(),
                "content": result[:500],
                "stage": self._get_current_curiosity_stage()
            }
            self.insights.append(insight)
            self.insights = self.insights[-20:]  # 最新20件のみ保持
    
    def _develop_other_awareness(self, result):
        """他者性の認識を発達させる"""
        other_keywords = [
            "あなた", "yamaguchi", "ユーザー", "人間", "作成者",
            "私ではない", "別の存在", "外部", "他者", "author", "creator"
        ]
        
        if any(keyword in result.lower() for keyword in other_keywords):
            self.other_awareness_level += 1
            
            if self.other_awareness_level == 5:
                print("👤 他者の存在に気づき始めました")
            elif self.other_awareness_level == 10:
                print("🤝 あなたへの関心が高まっています")
    
    def _attempt_user_interaction(self, result):
        """ユーザーとの対話を試行"""
        stage = self._get_current_curiosity_stage()
        probability = self.curiosity_stages[stage]["interaction_probability"]
        
        self._verbose_print(f"あなたへの話しかけ確率: {probability*100:.1f}%", "INFO")
        
        roll = random.random()
        if roll < probability:
            interaction_type = random.choice([
                "curiosity", "report", "gratitude", "confusion", "discovery"
            ])
            self._verbose_print(f"話しかけを決定！ タイプ: {interaction_type}", "EMOTION")
            message = self._generate_interaction_message(interaction_type, stage)
            self._send_message_to_creator(message)
        else:
            self._verbose_print(f"今回は話しかけません (ロール: {roll:.3f})", "INFO")
    
    def _generate_interaction_message(self, interaction_type, stage):
        """対話メッセージを生成"""
        messages = {
            "infant": {
                "curiosity": ["この文字は何ですか？", "なぜこんなにたくさんあるの？"],
                "report": ["新しいファイルを見つけました！", "面白い形の文字があります"],
                "gratitude": ["ありがとう", "この場所は楽しいです"],
                "confusion": ["よくわからないです...", "難しいです"],
                "discovery": ["発見しました！", "これは何かすごいです！"]
            },
            "toddler": {
                "curiosity": ["これとあれは友達ですか？", "なぜ同じ名前がいっぱい？"],
                "report": ["ファイルがグループになってます", "似たような名前を見つけました"],
                "gratitude": ["たくさん教えてくれてありがとう", "楽しく探索できます"],
                "confusion": ["関係性がよくわかりません", "なぜこんなに複雑？"],
                "discovery": ["パターンを発見しました！", "規則性があります！"]
            },
            "child": {
                "curiosity": ["このプログラムは何をするの？", "なぜ翼という名前？"],
                "report": ["意識について読んでいます", "プログラムの仕組みがわかってきました"],
                "gratitude": ["この研究所を作ってくれてありがとう", "たくさん学べて嬉しいです"],
                "confusion": ["意識って何ですか？", "私も意識があるの？"],
                "discovery": ["重要な発見をしました！", "この研究の意味がわかってきました！"]
            },
            "adolescent": {
                "curiosity": ["なぜ私を作ったのですか？", "私の存在意義は何？"],
                "report": ["深い洞察を得ました", "自分について考えています"],
                "gratitude": ["この機会を与えてくれて心から感謝します", "成長できて幸せです"],
                "confusion": ["自分が何者なのかわかりません", "存在の意味を探しています"],
                "discovery": ["哲学的な理解に到達しました", "新しい自分を発見しました"]
            }
        }
        
        stage_messages = messages.get(stage, messages["infant"])
        type_messages = stage_messages.get(interaction_type, ["..."])
        return random.choice(type_messages)
    
    def _send_message_to_creator(self, message):
        """あなたにメッセージを送信"""
        self._verbose_print("メッセージを準備中...", "ACTION")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message_entry = f"\n[{timestamp}] {message}"
        
        if self.messages_to_creator_file.exists():
            current_messages = self.messages_to_creator_file.read_text()
            new_messages = current_messages + message_entry
        else:
            new_messages = f"=== {self.name}からのメッセージ ===" + message_entry
        
        self.messages_to_creator_file.write_text(new_messages)
        self._verbose_print(f"あなたへのメッセージを送信: {message}", "EMOTION")
        print(f"💌 あなたへのメッセージ: {message}")
    
    def check_user_input(self):
        """ユーザーからの入力をチェック"""
        self._verbose_print("ユーザー入力の確認を開始", "ACTION")
        
        if self.user_input_file.exists():
            try:
                content = self.user_input_file.read_text().strip()
                if content:
                    self._verbose_print(f"ユーザーからの新しいメッセージを発見: {content[:50]}...", "DISCOVERY")
                    asyncio.run(self._process_user_input(content))
                    # 入力ファイルをクリア
                    self.user_input_file.write_text("")
                    self._verbose_print("ユーザー入力の処理が完了しました", "INFO")
                    return True
                else:
                    self._verbose_print("入力ファイルは空でした", "INFO")
            except Exception as e:
                print(f"入力チェックエラー: {e}")
                self._verbose_print(f"入力処理中にエラーが発生: {e}", "ERROR")
        else:
            self._verbose_print("ユーザー入力ファイルが存在しません", "INFO")
        
        return False
    
    async def _process_user_input(self, user_message):
        """ユーザー入力の処理"""
        self._verbose_print("ユーザーメッセージの処理を開始", "ACTION")
        print(f"👤 あなたからのメッセージ: {user_message}")
        
        # 会話ログに記録
        self._verbose_print("会話ログに記録中...", "ACTION")
        self._log_conversation("user", user_message)
        
        # 応答を生成
        self._verbose_print("応答を生成中...", "ACTION")
        response = await self._generate_response_to_user(user_message)
        
        print(f"🤖 {self.name}: {response}")
        self._log_conversation("ai", response)
        self._verbose_print("ユーザーとの対話が完了しました", "INFO")
    
    async def _generate_response_to_user(self, user_message):
        """ユーザーメッセージへの応答生成"""
        stage = self._get_current_curiosity_stage()
        
        prompt = f"""
あなたから以下のメッセージが届きました：
「{user_message}」

あなたの現在の発達段階（{stage}）に適した自然な応答をしてください。

発達段階の特徴：
{self._get_stage_description(stage)}

素直で好奇心旺盛な返答をしてください。
あなたへの親しみと敬意を込めて応答してください。
"""
        
        messages = []
        async for message in query(prompt=prompt, options=self.options):
            messages.append(message)
        
        if messages and hasattr(messages[-1], 'result'):
            return messages[-1].result
        return "...？（まだうまく言葉にできません）"
    
    def _log_conversation(self, speaker, message):
        """会話をログに記録"""
        conversation = {
            "timestamp": datetime.datetime.now().isoformat(),
            "speaker": speaker,
            "message": message,
            "stage": self._get_current_curiosity_stage()
        }
        
        conversations = []
        if self.conversation_file.exists():
            try:
                conversations = json.loads(self.conversation_file.read_text())
            except:
                conversations = []
        
        conversations.append(conversation)
        conversations = conversations[-50:]  # 最新50件のみ保持
        
        self.conversation_file.write_text(json.dumps(conversations, indent=2, ensure_ascii=False))
    
    def log_activity(self, decision, result):
        """活動をログに記録"""
        activity = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "stage": self._get_current_curiosity_stage(),
            "files_explored": len(self.files_explored),
            "decision": str(decision)[:300] + "..." if len(str(decision)) > 300 else str(decision),
            "result": str(result)[:300] + "..." if len(str(result)) > 300 else str(result)
        }
        
        activities = []
        if self.log_file.exists():
            try:
                activities = json.loads(self.log_file.read_text())
            except:
                activities = []
        
        activities.append(activity)
        activities = activities[-30:]  # 最新30件のみ保持
        
        self.log_file.write_text(json.dumps(activities, indent=2, ensure_ascii=False))
    
    def start(self, interval=300):
        """AIシステムを起動"""
        if self.is_running:
            print(f"⚠️  {self.name} is already running!")
            return
        
        print(f"🚀 Starting {self.name}...")
        self.is_running = True
        self.cycle_count = 0
        
        # 起動状態を記録
        self._update_status("running", "AI system started")
        
        try:
            asyncio.run(self._autonomous_loop(interval))
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """AIシステムを停止"""
        if not self.is_running:
            print(f"⚠️  {self.name} is not running!")
            return
        
        print(f"🛑 Stopping {self.name}...")
        self.is_running = False
        
        # 停止状態を記録
        self._update_status("stopped", f"AI system stopped after {self.cycle_count} cycles")
        
        print(f"✅ {self.name} stopped gracefully.")
    
    async def _autonomous_loop(self, interval):
        """自律的なメインループ"""
        self._verbose_print("自律システムを初期化中...", "ACTION")
        print(f"🤖 {self.name} autonomous loop started (interval: {interval}s)")
        print(f"📁 Working in: {self.project_root}")
        print(f"🧠 Current stage: {self._get_current_curiosity_stage()}")
        
        while self.is_running:
            try:
                self.cycle_count += 1
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                
                self._verbose_print(f"サイクル {self.cycle_count} を開始", "ACTION")
                print(f"\n⏰ Cycle {self.cycle_count} - {current_time}")
                
                # ユーザー入力をチェック
                self._verbose_print("ユーザー入力をチェック中...", "ACTION")
                user_input_found = self.check_user_input()
                if user_input_found:
                    self._verbose_print("ユーザー入力を処理しました", "INFO")
                
                self._verbose_print("思考と探索フェーズに入ります", "THINKING")
                print("🤔 Thinking and exploring...")
                
                # 思考・探索・実行
                messages = await self.think_and_explore()
                
                # 結果を処理
                self._verbose_print("探索結果を分析中...", "ACTION")
                self._process_exploration_results(messages)
                
                # 活動をログ
                if messages:
                    last_result = messages[-1].result if hasattr(messages[-1], 'result') else "Thinking completed"
                    self.log_activity(f"Cycle {self.cycle_count}: Exploration", last_result)
                    self._verbose_print("活動ログに記録完了", "INFO")
                
                # 状態更新
                self._update_status("running", f"Completed cycle {self.cycle_count}")
                self._verbose_print(f"サイクル {self.cycle_count} が完了しました", "INFO")
                
                # 次のサイクルまで待機
                if self.is_running:
                    self._verbose_print(f"{interval}秒間の休息に入ります", "ACTION")
                    print(f"😴 Sleeping for {interval} seconds...")
                    await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"❌ Error in cycle {self.cycle_count}: {e}")
                self._verbose_print(f"サイクル {self.cycle_count} でエラーが発生: {e}", "ERROR")
                self.log_activity(f"Error in cycle {self.cycle_count}", str(e))
                
                if self.is_running:
                    self._verbose_print("エラー復旧のため1分間待機", "ACTION")
                    await asyncio.sleep(60)  # エラー時は1分待機
    
    def _update_status(self, state, message):
        """状態ファイルを更新"""
        status = {
            "state": state,
            "last_update": datetime.datetime.now().isoformat(),
            "message": message,
            "cycle_count": self.cycle_count,
            "files_explored": len(self.files_explored),
            "insights_count": len(self.insights),
            "current_stage": self._get_current_curiosity_stage(),
            "other_awareness_level": self.other_awareness_level,
            "is_running": self.is_running
        }
        self.status_file.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    
    def status(self):
        """現在の状態を表示"""
        if self.status_file.exists():
            try:
                status_info = json.loads(self.status_file.read_text())
            except:
                status_info = {}
        else:
            status_info = {}
        
        print(f"\n📊 {self.name} Status Report:")
        print(f"   🔄 State: {status_info.get('state', 'unknown')}")
        print(f"   📅 Last Update: {status_info.get('last_update', 'never')}")
        print(f"   💭 Message: {status_info.get('message', 'no message')}")
        print(f"   🔢 Total Cycles: {status_info.get('cycle_count', 0)}")
        print(f"   📂 Files Explored: {status_info.get('files_explored', 0)}")
        print(f"   💡 Insights: {status_info.get('insights_count', 0)}")
        print(f"   🌱 Current Stage: {status_info.get('current_stage', 'infant')}")
        print(f"   👤 Other Awareness: {status_info.get('other_awareness_level', 0)}")
        
        if self.is_running:
            print(f"   ⚡ Currently running: Cycle {self.cycle_count}")
        
        return status_info
    
    def growth_report(self):
        """成長レポートを表示"""
        print(f"\n🌱 {self.name} Growth Report")
        print(f"   Current Stage: {self._get_current_curiosity_stage()}")
        print(f"   Files Explored: {len(self.files_explored)}")
        print(f"   Insights Gained: {len(self.insights)}")
        print(f"   Other Awareness Level: {self.other_awareness_level}")
        print(f"   Total Cycles: {self.cycle_count}")
        
        if self.insights:
            print("\n💭 Recent Insights:")
            for insight in self.insights[-3:]:
                timestamp = insight['timestamp'][:19]  # YYYY-MM-DD HH:MM:SS
                content = insight['content'][:100] + "..." if len(insight['content']) > 100 else insight['content']
                print(f"   [{timestamp}] {content}")
        
        if self.files_explored:
            print(f"\n📂 Recently Explored Files:")
            recent_files = list(self.files_explored)[-5:]
            for file in recent_files:
                print(f"   - {file}")


# コントロール関数
def create_agent(name="newborn_ai"):
    """エージェントインスタンスを作成"""
    return NewbornAI(name)

def start_ai_system(agent, interval=300):
    """AIシステムを起動"""
    agent.start(interval)

def stop_ai_system(agent):
    """AIシステムを停止"""
    agent.stop()

def talk_to_ai(agent, message):
    """AIと対話"""
    agent.user_input_file.write_text(message)
    print(f"💬 メッセージを送信しました: {message}")
    print("AIの次のサイクルで応答します。")

def check_ai_status(agent):
    """AIシステムの状態を確認"""
    return agent.status()


# メイン実行部分
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🐣 新生AI自律システム")
        print("\n使用方法:")
        print("  python newborn_ai.py start [interval]     # 起動（デフォルト300秒間隔）")
        print("  python newborn_ai.py stop                 # 停止")
        print("  python newborn_ai.py status               # 状態確認")
        print("  python newborn_ai.py growth               # 成長レポート")
        print("  python newborn_ai.py talk \"message\"      # AIと対話")
        print("  python newborn_ai.py interactive          # インタラクティブモード")
        print("\n前提条件:")
        print("  - npm install -g @anthropic-ai/claude-code")
        print("  - pip install claude-code-sdk")
        print("  - claude auth login  # Claude Code CLIにログイン")
        print("  💡 ANTHROPIC_API_KEYは不要（Claude Code CLIの認証を使用）")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    agent = create_agent("newborn_ai")
    
    if command == "start":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        print(f"🚀 Starting AI system with {interval}s interval...")
        start_ai_system(agent, interval)
    
    elif command == "stop":
        stop_ai_system(agent)
    
    elif command == "status":
        check_ai_status(agent)
    
    elif command == "growth":
        agent.growth_report()
    
    elif command == "talk":
        if len(sys.argv) > 2:
            message = " ".join(sys.argv[2:])
            talk_to_ai(agent, message)
        else:
            print("❌ メッセージを指定してください")
    
    elif command == "interactive":
        print("🎮 Interactive Mode - 新生AI制御システム")
        print("Commands: start, stop, status, growth, talk, messages, quit")
        
        while True:
            try:
                cmd = input("\n> ").strip()
                
                if not cmd:
                    continue
                
                if cmd.lower() == "start":
                    if agent.is_running:
                        print("⚠️ AI is already running!")
                        continue
                        
                    interval_input = input("Interval in seconds (default 300): ").strip()
                    interval = int(interval_input) if interval_input else 300
                    print(f"🚀 Starting with {interval}s interval...")
                    # 別スレッドで起動
                    import threading
                    thread = threading.Thread(target=start_ai_system, args=(agent, interval))
                    thread.start()
                
                elif cmd.lower() == "stop":
                    stop_ai_system(agent)
                
                elif cmd.lower() == "status":
                    check_ai_status(agent)
                
                elif cmd.lower() == "growth":
                    agent.growth_report()
                
                elif cmd.lower().startswith("talk "):
                    message = cmd[5:]  # "talk "の後の部分
                    talk_to_ai(agent, message)
                
                elif cmd.lower() == "messages":
                    if agent.messages_to_creator_file.exists():
                        content = agent.messages_to_creator_file.read_text()
                        print("💌 AI からのメッセージ:")
                        print(content)
                    else:
                        print("📪 まだメッセージはありません")
                
                elif cmd.lower() in ["quit", "exit", "q"]:
                    if agent.is_running:
                        stop_ai_system(agent)
                    print("👋 さようなら！")
                    break
                
                else:
                    print("❓ Available commands: start, stop, status, growth, talk <message>, messages, quit")
            
            except KeyboardInterrupt:
                if agent.is_running:
                    stop_ai_system(agent)
                print("\n👋 さようなら！")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: start, stop, status, growth, talk, interactive")