# NewbornAI 2.0 ローカル環境軽量セキュリティ仕様書

## 概要

ローカル環境でのclaude-code-sdk使用を前提とした、実用性を重視した軽量セキュリティフレームワークです。過度な複雑性を避けつつ、記憶データの適切な保護を実現します。

## セキュリティ境界の定義

### 1. 信頼境界モデル

```python
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

class TrustBoundary(Enum):
    """信頼境界レベル"""
    LOCAL_PROCESS = "local_process"      # 同一プロセス内
    LOCAL_MACHINE = "local_machine"      # 同一マシン内
    LOCAL_NETWORK = "local_network"      # ローカルネットワーク内
    EXTERNAL_API = "external_api"        # 外部APIサービス
    EXTERNAL_MCP = "external_mcp"        # 外部MCPサーバー

class DataSensitivity(Enum):
    """データ機密性レベル"""
    PUBLIC = "public"           # 公開可能
    INTERNAL = "internal"       # 内部使用のみ
    RESTRICTED = "restricted"   # 制限付き共有
    CONFIDENTIAL = "confidential"  # 機密情報

@dataclass
class SecurityContext:
    """セキュリティコンテキスト"""
    trust_boundary: TrustBoundary
    service_name: str
    service_type: str
    data_sensitivity_required: DataSensitivity
    user_consent: bool = False
    
class LocalSecurityManager:
    """ローカル環境セキュリティ管理"""
    
    def __init__(self):
        self.logger = logging.getLogger("local_security")
        
        # データ分類ルール
        self.data_classification = {
            # 公開可能データ
            'phi_summary': DataSensitivity.PUBLIC,
            'stage_level': DataSensitivity.PUBLIC,
            'development_progress': DataSensitivity.PUBLIC,
            
            # 内部使用データ
            'phi_detailed': DataSensitivity.INTERNAL,
            'stage_transitions': DataSensitivity.INTERNAL,
            'qualitative_metrics': DataSensitivity.INTERNAL,
            
            # 制限付きデータ
            'experience_summaries': DataSensitivity.RESTRICTED,
            'temporal_patterns': DataSensitivity.RESTRICTED,
            'behavioral_analytics': DataSensitivity.RESTRICTED,
            
            # 機密データ
            'detailed_memories': DataSensitivity.CONFIDENTIAL,
            'personal_experiences': DataSensitivity.CONFIDENTIAL,
            'temporal_consciousness_raw': DataSensitivity.CONFIDENTIAL,
            'phi_calculation_internals': DataSensitivity.CONFIDENTIAL
        }
        
        # サービス別信頼レベル
        self.service_trust_levels = {
            'localhost': TrustBoundary.LOCAL_MACHINE,
            'anthropic_api': TrustBoundary.EXTERNAL_API,
            'local_mcp_server': TrustBoundary.LOCAL_MACHINE,
            'visualization_tools': TrustBoundary.LOCAL_NETWORK,
            'external_mcp_server': TrustBoundary.EXTERNAL_MCP
        }
```

### 2. データフィルタリングシステム

```python
class DataFilter:
    """データフィルタリング"""
    
    def __init__(self, security_manager: LocalSecurityManager):
        self.security_manager = security_manager
        self.logger = logging.getLogger("data_filter")
        
    def filter_for_context(
        self, 
        data: Dict[str, Any], 
        context: SecurityContext
    ) -> Dict[str, Any]:
        """コンテキストに応じたデータフィルタリング"""
        
        filtered_data = {}
        
        for key, value in data.items():
            data_sensitivity = self.security_manager.data_classification.get(
                key, DataSensitivity.CONFIDENTIAL
            )
            
            if self.is_access_allowed(data_sensitivity, context):
                filtered_data[key] = self.transform_data_for_context(
                    key, value, context
                )
            else:
                self.logger.debug(
                    f"Filtered out {key} (sensitivity: {data_sensitivity.value}) "
                    f"for context {context.service_name}"
                )
                
        return filtered_data
        
    def is_access_allowed(
        self, 
        data_sensitivity: DataSensitivity, 
        context: SecurityContext
    ) -> bool:
        """アクセス許可判定"""
        
        # ローカルプロセス内は全てアクセス可能
        if context.trust_boundary == TrustBoundary.LOCAL_PROCESS:
            return True
            
        # ローカルマシン内は機密データ以外アクセス可能
        if context.trust_boundary == TrustBoundary.LOCAL_MACHINE:
            return data_sensitivity != DataSensitivity.CONFIDENTIAL
            
        # 外部APIは公開データのみ（ユーザー同意があれば内部データも可）
        if context.trust_boundary == TrustBoundary.EXTERNAL_API:
            if data_sensitivity == DataSensitivity.PUBLIC:
                return True
            elif data_sensitivity == DataSensitivity.INTERNAL and context.user_consent:
                return True
            else:
                return False
                
        # 外部MCPサーバーは公開データのみ
        if context.trust_boundary == TrustBoundary.EXTERNAL_MCP:
            return data_sensitivity == DataSensitivity.PUBLIC
            
        return False
        
    def transform_data_for_context(
        self, 
        key: str, 
        value: Any, 
        context: SecurityContext
    ) -> Any:
        """コンテキスト別データ変換"""
        
        # 外部サービス向けは数値データを丸める
        if context.trust_boundary in [TrustBoundary.EXTERNAL_API, TrustBoundary.EXTERNAL_MCP]:
            if isinstance(value, float):
                # φ値は小数点2桁まで
                if 'phi' in key.lower():
                    return round(value, 2)
                # その他の数値は小数点1桁まで
                return round(value, 1)
                
        # 可視化ツール向けは配列データを間引く
        if context.service_type == 'visualization' and isinstance(value, list):
            if len(value) > 100:
                # 100要素を超える配列は間引き
                step = len(value) // 100
                return value[::step]
                
        return value
```

### 3. ユーザー同意管理

```python
class UserConsentManager:
    """ユーザー同意管理"""
    
    def __init__(self):
        self.consent_cache: Dict[str, Dict[str, bool]] = {}
        self.consent_expiry: Dict[str, float] = {}
        self.logger = logging.getLogger("user_consent")
        
    async def request_consent(
        self, 
        service_name: str, 
        data_types: List[str],
        purpose: str
    ) -> bool:
        """ユーザー同意リクエスト"""
        
        # キャッシュ確認
        cache_key = f"{service_name}:{':'.join(sorted(data_types))}"
        if self.is_consent_valid(cache_key):
            return self.consent_cache[service_name].get(cache_key, False)
            
        # ユーザーに同意確認
        consent_granted = await self.prompt_user_consent(
            service_name, data_types, purpose
        )
        
        # 同意結果をキャッシュ（1時間有効）\n        import time\n        if service_name not in self.consent_cache:\n            self.consent_cache[service_name] = {}\n            \n        self.consent_cache[service_name][cache_key] = consent_granted\n        self.consent_expiry[cache_key] = time.time() + 3600  # 1時間\n        \n        return consent_granted\n        \n    async def prompt_user_consent(\n        self, \n        service_name: str, \n        data_types: List[str],\n        purpose: str\n    ) -> bool:\n        \"\"\"ユーザー同意プロンプト\"\"\"\n        \n        print(f\"\\n=== データ共有の同意確認 ===\")\n        print(f\"サービス: {service_name}\")\n        print(f\"目的: {purpose}\")\n        print(f\"共有データ: {', '.join(data_types)}\")\n        print(\"\\n以下のデータを共有することに同意しますか？\")\n        \n        # 実際の実装では適切なUI/CLIプロンプトを使用\n        response = input(\"同意する場合は 'yes' を入力してください: \")\n        \n        consent = response.lower() in ['yes', 'y', '同意']\n        \n        if consent:\n            self.logger.info(f\"User granted consent for {service_name}: {data_types}\")\n        else:\n            self.logger.info(f\"User denied consent for {service_name}: {data_types}\")\n            \n        return consent\n        \n    def is_consent_valid(self, cache_key: str) -> bool:\n        \"\"\"同意の有効性確認\"\"\"\n        import time\n        \n        if cache_key not in self.consent_expiry:\n            return False\n            \n        return time.time() < self.consent_expiry[cache_key]\n        \n    def revoke_consent(self, service_name: str):\n        \"\"\"同意の取り消し\"\"\"\n        if service_name in self.consent_cache:\n            del self.consent_cache[service_name]\n            \n        # 期限切れエントリを削除\n        expired_keys = [\n            key for key, expiry in self.consent_expiry.items()\n            if service_name in key\n        ]\n        \n        for key in expired_keys:\n            del self.consent_expiry[key]\n            \n        self.logger.info(f\"Revoked all consents for {service_name}\")\n```\n\n### 4. セキュリティ監査システム\n\n```python\nimport json\nfrom datetime import datetime\nfrom pathlib import Path\n\nclass SecurityAuditor:\n    \"\"\"セキュリティ監査\"\"\"\n    \n    def __init__(self, audit_log_path: str = \"./logs/security_audit.jsonl\"):\n        self.audit_log_path = Path(audit_log_path)\n        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)\n        self.logger = logging.getLogger(\"security_auditor\")\n        \n    async def log_data_access(\n        self,\n        context: SecurityContext,\n        data_keys: List[str],\n        access_granted: bool,\n        user_consent: bool = False\n    ):\n        \"\"\"データアクセスログ\"\"\"\n        \n        audit_entry = {\n            'timestamp': datetime.utcnow().isoformat(),\n            'event_type': 'data_access',\n            'service_name': context.service_name,\n            'service_type': context.service_type,\n            'trust_boundary': context.trust_boundary.value,\n            'data_keys': data_keys,\n            'access_granted': access_granted,\n            'user_consent': user_consent,\n            'sensitivity_required': context.data_sensitivity_required.value\n        }\n        \n        await self.write_audit_log(audit_entry)\n        \n    async def log_consent_event(\n        self,\n        service_name: str,\n        data_types: List[str],\n        consent_granted: bool,\n        purpose: str\n    ):\n        \"\"\"同意イベントログ\"\"\"\n        \n        audit_entry = {\n            'timestamp': datetime.utcnow().isoformat(),\n            'event_type': 'user_consent',\n            'service_name': service_name,\n            'data_types': data_types,\n            'consent_granted': consent_granted,\n            'purpose': purpose\n        }\n        \n        await self.write_audit_log(audit_entry)\n        \n    async def log_security_violation(\n        self,\n        violation_type: str,\n        details: Dict[str, Any]\n    ):\n        \"\"\"セキュリティ違反ログ\"\"\"\n        \n        audit_entry = {\n            'timestamp': datetime.utcnow().isoformat(),\n            'event_type': 'security_violation',\n            'violation_type': violation_type,\n            'details': details,\n            'severity': 'HIGH'\n        }\n        \n        await self.write_audit_log(audit_entry)\n        \n        # 重要な違反は即座にログ出力\n        self.logger.warning(f\"Security violation: {violation_type} - {details}\")\n        \n    async def write_audit_log(self, entry: Dict[str, Any]):\n        \"\"\"監査ログ書き込み\"\"\"\n        \n        try:\n            with open(self.audit_log_path, 'a', encoding='utf-8') as f:\n                f.write(json.dumps(entry, ensure_ascii=False) + '\\n')\n        except Exception as e:\n            self.logger.error(f\"Failed to write audit log: {e}\")\n            \n    async def generate_security_report(self, days: int = 7) -> Dict[str, Any]:\n        \"\"\"セキュリティレポート生成\"\"\"\n        \n        import time\n        from collections import defaultdict\n        \n        cutoff_time = time.time() - (days * 24 * 3600)\n        \n        stats = {\n            'total_accesses': 0,\n            'denied_accesses': 0,\n            'services': defaultdict(int),\n            'data_types': defaultdict(int),\n            'violations': []\n        }\n        \n        try:\n            with open(self.audit_log_path, 'r', encoding='utf-8') as f:\n                for line in f:\n                    try:\n                        entry = json.loads(line.strip())\n                        entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()\n                        \n                        if entry_time < cutoff_time:\n                            continue\n                            \n                        if entry['event_type'] == 'data_access':\n                            stats['total_accesses'] += 1\n                            if not entry['access_granted']:\n                                stats['denied_accesses'] += 1\n                                \n                            stats['services'][entry['service_name']] += 1\n                            \n                            for data_key in entry['data_keys']:\n                                stats['data_types'][data_key] += 1\n                                \n                        elif entry['event_type'] == 'security_violation':\n                            stats['violations'].append(entry)\n                            \n                    except json.JSONDecodeError:\n                        continue\n                        \n        except FileNotFoundError:\n            pass\n            \n        return dict(stats)\n```\n\n### 5. 統合セキュリティフレームワーク\n\n```python\nclass LightweightSecurityFramework:\n    \"\"\"軽量セキュリティフレームワーク統合\"\"\"\n    \n    def __init__(self):\n        self.security_manager = LocalSecurityManager()\n        self.data_filter = DataFilter(self.security_manager)\n        self.consent_manager = UserConsentManager()\n        self.auditor = SecurityAuditor()\n        self.logger = logging.getLogger(\"security_framework\")\n        \n    async def secure_data_access(\n        self,\n        data: Dict[str, Any],\n        service_name: str,\n        service_type: str,\n        purpose: str = \"データ処理\"\n    ) -> Dict[str, Any]:\n        \"\"\"セキュアなデータアクセス\"\"\"\n        \n        # サービスの信頼境界判定\n        trust_boundary = self.security_manager.service_trust_levels.get(\n            service_name, TrustBoundary.EXTERNAL_MCP\n        )\n        \n        # セキュリティコンテキスト作成\n        context = SecurityContext(\n            trust_boundary=trust_boundary,\n            service_name=service_name,\n            service_type=service_type,\n            data_sensitivity_required=DataSensitivity.INTERNAL\n        )\n        \n        # 外部サービスの場合はユーザー同意確認\n        if trust_boundary in [TrustBoundary.EXTERNAL_API, TrustBoundary.EXTERNAL_MCP]:\n            data_types = list(data.keys())\n            context.user_consent = await self.consent_manager.request_consent(\n                service_name, data_types, purpose\n            )\n            \n        # データフィルタリング\n        filtered_data = self.data_filter.filter_for_context(data, context)\n        \n        # 監査ログ記録\n        await self.auditor.log_data_access(\n            context,\n            list(data.keys()),\n            len(filtered_data) > 0,\n            context.user_consent\n        )\n        \n        return filtered_data\n        \n    async def register_service(\n        self,\n        service_name: str,\n        trust_level: TrustBoundary,\n        capabilities: List[str]\n    ):\n        \"\"\"サービス登録\"\"\"\n        \n        self.security_manager.service_trust_levels[service_name] = trust_level\n        \n        self.logger.info(\n            f\"Registered service {service_name} with trust level {trust_level.value}\"\n        )\n        \n    async def security_health_check(self) -> Dict[str, Any]:\n        \"\"\"セキュリティヘルスチェック\"\"\"\n        \n        report = await self.auditor.generate_security_report(days=1)\n        \n        health_status = {\n            'status': 'healthy',\n            'issues': [],\n            'recommendations': []\n        }\n        \n        # 異常なアクセス拒否率をチェック\n        if report['total_accesses'] > 0:\n            denial_rate = report['denied_accesses'] / report['total_accesses']\n            if denial_rate > 0.5:  # 50%以上の拒否率\n                health_status['issues'].append(\n                    f\"High access denial rate: {denial_rate:.1%}\"\n                )\n                health_status['status'] = 'warning'\n                \n        # セキュリティ違反をチェック\n        if report['violations']:\n            health_status['issues'].append(\n                f\"{len(report['violations'])} security violations detected\"\n            )\n            health_status['status'] = 'critical'\n            \n        return health_status\n```\n\n### 6. 使用例\n\n```python\n# セキュリティフレームワーク初期化\nsecurity = LightweightSecurityFramework()\n\n# サービス登録\nawait security.register_service(\n    \"photoshop_plugin\", \n    TrustBoundary.LOCAL_NETWORK,\n    [\"visualization\", \"image_editing\"]\n)\n\nawait security.register_service(\n    \"anthropic_api\",\n    TrustBoundary.EXTERNAL_API,\n    [\"text_processing\", \"analysis\"]\n)\n\n# 意識データの安全な共有\nconsciousness_data = {\n    'phi_summary': 25.5,\n    'stage_level': 3,\n    'detailed_memories': [...],  # 機密データ\n    'experience_summaries': [...],  # 制限付きデータ\n}\n\n# Photoshopプラグインには可視化データのみ\nfiltered_for_photoshop = await security.secure_data_access(\n    consciousness_data,\n    \"photoshop_plugin\",\n    \"visualization\",\n    \"意識状態の視覚化\"\n)\n\n# Anthropic APIには匿名化されたデータのみ（ユーザー同意必要）\nfiltered_for_api = await security.secure_data_access(\n    consciousness_data,\n    \"anthropic_api\",\n    \"analysis\",\n    \"意識パターンの分析\"\n)\n\n# セキュリティ状況確認\nhealth = await security.security_health_check()\nprint(f\"Security status: {health['status']}\")\n```\n\nこの軽量セキュリティフレームワークにより、ローカル環境での実用性を保ちながら、記憶データの適切な保護を実現できます。