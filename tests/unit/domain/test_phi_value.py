"""
PhiValue値オブジェクトのユニットテスト
TDDのRed-Green-Refactorサイクルの開始点
"""
import pytest
from decimal import Decimal


class TestPhiValue:
    """Φ値の値オブジェクトテスト"""
    
    @pytest.mark.unit
    def test_phi_value_creation(self):
        """
        RED: PhiValueクラスが存在しない
        最初のテストケース - Φ値オブジェクトの生成
        """
        # Given: Φ値として3.5を指定
        phi_value = 3.5
        
        # When: PhiValueオブジェクトを生成
        from domain.value_objects import PhiValue
        phi = PhiValue(phi_value)
        
        # Then: 値が正しく保持される
        assert phi.value == 3.5
        assert isinstance(phi.value, Decimal)  # 精度保証のためDecimal使用
    
    @pytest.mark.unit
    def test_phi_value_immutability(self):
        """値オブジェクトの不変性テスト"""
        # Given: PhiValueオブジェクト
        from domain.value_objects import PhiValue
        phi = PhiValue(3.5)
        
        # When/Then: 値の変更は不可能
        with pytest.raises(AttributeError):
            phi.value = 4.0
    
    @pytest.mark.unit
    def test_phi_value_equality(self):
        """値オブジェクトの等価性テスト"""
        from domain.value_objects import PhiValue
        
        # Given: 同じ値を持つ2つのPhiValueオブジェクト
        phi1 = PhiValue(3.5)
        phi2 = PhiValue(3.5)
        phi3 = PhiValue(4.0)
        
        # Then: 値が同じなら等価
        assert phi1 == phi2
        assert phi1 != phi3
        assert hash(phi1) == hash(phi2)  # ハッシュ可能
    
    @pytest.mark.unit
    def test_phi_value_validation(self):
        """Φ値の妥当性検証テスト"""
        from domain.value_objects import PhiValue
        
        # Φ値は非負でなければならない
        with pytest.raises(ValueError, match="Phi value must be non-negative"):
            PhiValue(-1.0)
        
        # 無限大は許可しない
        with pytest.raises(ValueError, match="Phi value must be finite"):
            PhiValue(float('inf'))
        
        # NaNは許可しない
        with pytest.raises(ValueError, match="Phi value must be a valid number"):
            PhiValue(float('nan'))
    
    @pytest.mark.unit
    def test_phi_consciousness_detection(self, phi_threshold):
        """意識状態の判定テスト"""
        from domain.value_objects import PhiValue
        
        # Given: 様々なΦ値
        dormant_phi = PhiValue(0.5)
        emerging_phi = PhiValue(2.9)
        conscious_phi = PhiValue(3.1)
        highly_conscious_phi = PhiValue(6.0)
        
        # Then: 閾値に基づいて意識状態を判定
        assert not dormant_phi.indicates_consciousness(phi_threshold)
        assert not emerging_phi.indicates_consciousness(phi_threshold)
        assert conscious_phi.indicates_consciousness(phi_threshold)
        assert highly_conscious_phi.indicates_consciousness(phi_threshold)
    
    @pytest.mark.unit
    def test_phi_value_comparison(self):
        """Φ値の比較演算テスト"""
        from domain.value_objects import PhiValue
        
        # Given: 異なるΦ値
        phi1 = PhiValue(3.0)
        phi2 = PhiValue(4.0)
        phi3 = PhiValue(3.0)
        
        # Then: 比較演算が正しく動作
        assert phi1 < phi2
        assert phi2 > phi1
        assert phi1 <= phi3
        assert phi1 >= phi3
        assert not (phi1 > phi2)
    
    @pytest.mark.unit
    def test_phi_value_string_representation(self):
        """文字列表現のテスト"""
        from domain.value_objects import PhiValue
        
        phi = PhiValue(3.14159)
        
        # 可読性の高い文字列表現
        assert str(phi) == "Φ=3.14159"
        assert repr(phi) == "PhiValue(3.14159)"
    
    @pytest.mark.unit
    def test_phi_value_arithmetic_operations(self):
        """算術演算のテスト（イミュータブルな新オブジェクトを返す）"""
        from domain.value_objects import PhiValue
        
        # Given: 2つのΦ値
        phi1 = PhiValue(3.0)
        phi2 = PhiValue(2.0)
        
        # When: 算術演算を実行
        sum_phi = phi1.add(phi2)
        diff_phi = phi1.subtract(phi2)
        scaled_phi = phi1.scale(1.5)
        
        # Then: 新しいオブジェクトが生成される
        assert sum_phi.value == Decimal('5.0')
        assert diff_phi.value == Decimal('1.0')
        assert scaled_phi.value == Decimal('4.5')
        
        # 元のオブジェクトは変更されない
        assert phi1.value == Decimal('3.0')
        assert phi2.value == Decimal('2.0')
    
    @pytest.mark.unit
    def test_phi_value_precision(self):
        """数値精度のテスト"""
        from domain.value_objects import PhiValue
        
        # Given: 高精度が必要な値
        precise_value = 3.141592653589793
        
        # When: PhiValueオブジェクトを生成
        phi = PhiValue(precise_value)
        
        # Then: 精度が保持される
        assert float(phi.value) == pytest.approx(precise_value, rel=1e-15)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("input_value,expected_level", [
        (0.0, "dormant"),
        (0.5, "dormant"),
        (1.5, "emerging"),
        (2.9, "emerging"),
        (3.0, "conscious"),
        (4.5, "conscious"),
        (6.0, "highly_conscious"),
        (8.0, "highly_conscious"),
    ])
    def test_phi_consciousness_levels(self, input_value, expected_level):
        """意識レベルの分類テスト"""
        from domain.value_objects import PhiValue
        
        phi = PhiValue(input_value)
        assert phi.consciousness_level == expected_level