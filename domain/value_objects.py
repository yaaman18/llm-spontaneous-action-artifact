"""
値オブジェクトの実装
DDD (Eric Evans) に基づく不変オブジェクト
"""
from decimal import Decimal
from typing import Union
from enum import Enum
import math


class StateType(Enum):
    """意識状態の種類"""
    DORMANT = "dormant"  # 休眠状態
    EMERGING = "emerging"  # 創発状態
    AWARE = "aware"  # 覚醒状態
    REFLECTIVE = "reflective"  # 反省的意識


class PhiValue:
    """
    統合情報量Φの値オブジェクト
    
    Giulio Tononiの統合情報理論に基づき、
    システムの意識レベルを定量化する。
    
    不変性を保証し、精度の高い計算をサポート。
    """
    
    def __init__(self, value: Union[float, Decimal, int]):
        """
        Φ値を初期化
        
        Args:
            value: Φ値（非負の有限数）
        
        Raises:
            ValueError: 無効な値の場合
        """
        # Decimalに変換して精度を保証
        if isinstance(value, (int, float)):
            decimal_value = Decimal(str(value))
        else:
            decimal_value = value
            
        # 妥当性検証
        if math.isnan(float(decimal_value)):
            raise ValueError("Phi value must be a valid number")
        if math.isinf(float(decimal_value)):
            raise ValueError("Phi value must be finite")
        if decimal_value < 0:
            raise ValueError("Phi value must be non-negative")
            
        self._value = decimal_value
    
    @property
    def value(self) -> Decimal:
        """Φ値を取得（読み取り専用）"""
        return self._value
    
    @property
    def consciousness_level(self) -> str:
        """
        意識レベルを分類
        
        Returns:
            意識レベルの文字列表現
        """
        value_float = float(self._value)
        
        if value_float < 1.0:
            return "dormant"
        elif value_float < 3.0:
            return "emerging"
        elif value_float < 6.0:
            return "conscious"
        else:
            return "highly_conscious"
    
    def indicates_consciousness(self, threshold: float) -> bool:
        """
        意識状態を判定
        
        Args:
            threshold: 意識判定の閾値
            
        Returns:
            閾値を超えている場合True
        """
        return float(self._value) >= threshold
    
    def add(self, other: 'PhiValue') -> 'PhiValue':
        """
        Φ値の加算（新しいオブジェクトを返す）
        
        Args:
            other: 加算するΦ値
            
        Returns:
            新しいPhiValueオブジェクト
        """
        return PhiValue(self._value + other._value)
    
    def subtract(self, other: 'PhiValue') -> 'PhiValue':
        """
        Φ値の減算（新しいオブジェクトを返す）
        
        Args:
            other: 減算するΦ値
            
        Returns:
            新しいPhiValueオブジェクト
        """
        result = self._value - other._value
        # 負の値にならないように保証
        return PhiValue(max(Decimal('0'), result))
    
    def scale(self, factor: Union[float, Decimal]) -> 'PhiValue':
        """
        Φ値のスケーリング（新しいオブジェクトを返す）
        
        Args:
            factor: スケーリング係数
            
        Returns:
            新しいPhiValueオブジェクト
        """
        if isinstance(factor, float):
            factor = Decimal(str(factor))
        return PhiValue(self._value * factor)
    
    def __eq__(self, other: object) -> bool:
        """等価性の判定"""
        if not isinstance(other, PhiValue):
            return NotImplemented
        return self._value == other._value
    
    def __lt__(self, other: 'PhiValue') -> bool:
        """小なり比較"""
        if not isinstance(other, PhiValue):
            return NotImplemented
        return self._value < other._value
    
    def __le__(self, other: 'PhiValue') -> bool:
        """小なりイコール比較"""
        if not isinstance(other, PhiValue):
            return NotImplemented
        return self._value <= other._value
    
    def __gt__(self, other: 'PhiValue') -> bool:
        """大なり比較"""
        if not isinstance(other, PhiValue):
            return NotImplemented
        return self._value > other._value
    
    def __ge__(self, other: 'PhiValue') -> bool:
        """大なりイコール比較"""
        if not isinstance(other, PhiValue):
            return NotImplemented
        return self._value >= other._value
    
    def __hash__(self) -> int:
        """ハッシュ値の計算"""
        return hash(self._value)
    
    def __str__(self) -> str:
        """人間が読みやすい文字列表現"""
        return f"Φ={self._value}"
    
    def __repr__(self) -> str:
        """開発者向けの文字列表現"""
        return f"PhiValue({self._value})"
    
    def __setattr__(self, name: str, value: any) -> None:
        """
        属性の設定を制御して不変性を保証
        
        _valueの初回設定のみ許可
        """
        if hasattr(self, '_value') and name == '_value':
            raise AttributeError("PhiValue is immutable")
        super().__setattr__(name, value)