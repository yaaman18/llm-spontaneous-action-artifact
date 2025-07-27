"""
計算結果のキャッシング戦略
Martin Fowlerのリファクタリング原則に基づく設計
"""
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from collections import OrderedDict
import threading

from .value_objects import PhiValue


@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    value: Any
    timestamp: datetime
    hit_count: int = 0
    
    def is_expired(self, ttl: timedelta) -> bool:
        """有効期限切れかどうか"""
        return datetime.now() - self.timestamp > ttl


class PhiCalculationCache:
    """Φ値計算結果のキャッシュ"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl: timedelta = timedelta(minutes=15),
                 eviction_policy: str = 'lru'):
        """
        Args:
            max_size: 最大キャッシュサイズ
            ttl: キャッシュの有効期限
            eviction_policy: 退避ポリシー ('lru', 'lfu', 'fifo')
        """
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_policy = eviction_policy
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def _generate_key(self, connectivity: Any, state: Any) -> str:
        """キャッシュキーを生成"""
        # NumPy配列を含む任意のデータからハッシュを生成
        data = {
            'connectivity': connectivity.tolist() if hasattr(connectivity, 'tolist') else str(connectivity),
            'state': state.tolist() if hasattr(state, 'tolist') else str(state)
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def get(self, connectivity: Any, state: Any) -> Optional[PhiValue]:
        """キャッシュから値を取得"""
        key = self._generate_key(connectivity, state)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # 有効期限チェック
                if entry.is_expired(self.ttl):
                    del self._cache[key]
                    self._stats['misses'] += 1
                    return None
                
                # ヒット処理
                entry.hit_count += 1
                self._stats['hits'] += 1
                
                # LRUの場合は最後に移動
                if self.eviction_policy == 'lru':
                    self._cache.move_to_end(key)
                
                return entry.value
            
            self._stats['misses'] += 1
            return None
    
    def put(self, connectivity: Any, state: Any, value: PhiValue) -> None:
        """キャッシュに値を格納"""
        key = self._generate_key(connectivity, state)
        
        with self._lock:
            # 既存エントリの更新
            if key in self._cache:
                self._cache[key] = CacheEntry(value, datetime.now())
                if self.eviction_policy == 'lru':
                    self._cache.move_to_end(key)
                return
            
            # 容量チェックと退避
            if len(self._cache) >= self.max_size:
                self._evict()
            
            # 新規エントリの追加
            self._cache[key] = CacheEntry(value, datetime.now())
    
    def _evict(self) -> None:
        """退避ポリシーに基づいてエントリを削除"""
        if not self._cache:
            return
        
        if self.eviction_policy == 'lru':
            # 最も古いエントリを削除
            self._cache.popitem(last=False)
        elif self.eviction_policy == 'lfu':
            # 最も使用頻度の低いエントリを削除
            min_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].hit_count)
            del self._cache[min_key]
        else:  # fifo
            # 最初のエントリを削除
            self._cache.popitem(last=False)
        
        self._stats['evictions'] += 1
    
    def clear(self) -> None:
        """キャッシュをクリア"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計を取得"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': hit_rate
            }


class CachedPhiCalculator:
    """キャッシュ機能付きΦ値計算器"""
    
    def __init__(self, 
                 calculator: Callable,
                 cache: Optional[PhiCalculationCache] = None):
        """
        Args:
            calculator: 実際の計算を行う関数
            cache: キャッシュインスタンス
        """
        self.calculator = calculator
        self.cache = cache or PhiCalculationCache()
    
    def calculate(self, connectivity: Any, state: Any) -> PhiValue:
        """キャッシュを利用したΦ値計算"""
        # キャッシュから取得を試みる
        cached_value = self.cache.get(connectivity, state)
        if cached_value is not None:
            return cached_value
        
        # キャッシュミスの場合は計算
        result = self.calculator(connectivity, state)
        
        # 結果をキャッシュに格納
        self.cache.put(connectivity, state, result)
        
        return result