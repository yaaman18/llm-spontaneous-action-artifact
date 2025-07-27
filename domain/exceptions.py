"""
ドメイン例外の定義
ビジネスルール違反を表現する例外クラス
"""


class DomainException(Exception):
    """すべてのドメイン例外の基底クラス"""
    pass


class InvalidStateTransition(DomainException):
    """無効な状態遷移を表す例外"""
    pass


class PhiValueOutOfRange(DomainException):
    """Φ値が有効範囲外の場合の例外"""
    pass


class ConsciousnessEmergenceFailed(DomainException):
    """意識創発の失敗を表す例外"""
    pass


class InsufficientIntegration(DomainException):
    """統合が不十分な場合の例外"""
    pass


class TemporalCoherenceLost(DomainException):
    """時間的一貫性が失われた場合の例外"""
    pass


class SubsystemBoundaryViolation(DomainException):
    """サブシステム境界の違反を表す例外"""
    pass