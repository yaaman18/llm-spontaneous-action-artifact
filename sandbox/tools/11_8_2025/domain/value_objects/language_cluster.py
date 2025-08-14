"""
Language Cluster Value Object.

A value object representing a language cluster with immutable characteristics
and behavior for multilingual text processing. Follows DDD value object
principles with equality based on value rather than identity.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import hashlib
import json
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class LanguageCluster:
    """
    Immutable value object representing a language cluster.
    
    A language cluster encapsulates the statistical and linguistic
    characteristics of a specific language or language variant
    discovered through unsupervised learning.
    
    Value Object Principles:
    - Immutable after creation
    - Equality based on values, not identity  
    - No side effects
    - Self-validating
    """
    
    cluster_id: str
    character_statistics: Dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.7
    pattern_signatures: Dict[str, float] = field(default_factory=dict)
    creation_timestamp: Optional[float] = None
    sample_count: int = 0
    
    def __post_init__(self):
        """Validate value object constraints after initialization."""
        if not self.cluster_id:
            raise ValueError("cluster_id cannot be empty")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.sample_count < 0:
            raise ValueError("sample_count cannot be negative")
    
    def extract_features(self) -> Dict[str, Any]:
        """
        Extract feature vector from cluster characteristics.
        
        Returns:
            Dictionary containing feature values for similarity computation
        """
        features = {}
        
        # Character-based features
        features.update(self.character_statistics)
        
        # Pattern-based features  
        features.update(self.pattern_signatures)
        
        # Meta features
        features['confidence_threshold'] = self.confidence_threshold
        features['sample_count'] = self.sample_count
        
        return features
    
    def compute_similarity(self, other: 'LanguageCluster') -> float:
        """
        Compute similarity with another language cluster.
        
        Args:
            other: Another LanguageCluster to compare with
            
        Returns:
            Similarity score between 0.0 and 1.0
            
        Raises:
            ValueError: If other is not a LanguageCluster
        """
        if not isinstance(other, LanguageCluster):
            raise ValueError("Can only compare with another LanguageCluster")
        
        self_features = self.extract_features()
        other_features = other.extract_features()
        
        # Get common keys
        common_keys = set(self_features.keys()) & set(other_features.keys())
        
        if not common_keys:
            return 0.0
        
        # Compute cosine similarity for numeric features
        similarity_scores = []
        for key in common_keys:
            self_val = self_features[key]
            other_val = other_features[key]
            
            # Handle different data types
            if isinstance(self_val, bool) and isinstance(other_val, bool):
                similarity_scores.append(1.0 if self_val == other_val else 0.0)
            elif isinstance(self_val, (int, float)) and isinstance(other_val, (int, float)):
                # Normalized difference for numeric values
                max_val = max(abs(self_val), abs(other_val), 1e-8)
                diff = abs(self_val - other_val) / max_val
                similarity_scores.append(max(0.0, 1.0 - diff))
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    def get_signature_hash(self) -> str:
        """
        Generate a hash signature for the cluster.
        
        Returns:
            Hexadecimal hash string representing cluster signature
        """
        # Create deterministic representation
        signature_data = {
            'cluster_id': self.cluster_id,
            'character_statistics': sorted(self.character_statistics.items()),
            'confidence_threshold': self.confidence_threshold,
            'pattern_signatures': sorted(self.pattern_signatures.items())
        }
        
        # Generate hash
        signature_json = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_json.encode()).hexdigest()[:16]
    
    def with_updated_statistics(self, new_statistics: Dict[str, Any]) -> 'LanguageCluster':
        """
        Create a new cluster with updated statistics.
        
        Since this is an immutable value object, returns a new instance
        with modified statistics while preserving other attributes.
        
        Args:
            new_statistics: Updated character statistics
            
        Returns:
            New LanguageCluster instance with updated statistics
        """
        from dataclasses import replace
        return replace(self, character_statistics=new_statistics)
    
    def with_updated_confidence(self, new_confidence: float) -> 'LanguageCluster':
        """
        Create a new cluster with updated confidence threshold.
        
        Args:
            new_confidence: New confidence threshold (0.0 to 1.0)
            
        Returns:
            New LanguageCluster instance with updated confidence
            
        Raises:
            ValueError: If confidence is not in valid range
        """
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        from dataclasses import replace
        return replace(self, confidence_threshold=new_confidence)
    
    def is_similar_to(self, other: 'LanguageCluster', threshold: float = 0.8) -> bool:
        """
        Check if this cluster is similar to another cluster.
        
        Args:
            other: Another LanguageCluster to compare with
            threshold: Similarity threshold for comparison
            
        Returns:
            True if similarity exceeds threshold, False otherwise
        """
        similarity = self.compute_similarity(other)
        return similarity >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert cluster to dictionary representation.
        
        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            'cluster_id': self.cluster_id,
            'character_statistics': self.character_statistics,
            'confidence_threshold': self.confidence_threshold,
            'pattern_signatures': self.pattern_signatures,
            'creation_timestamp': self.creation_timestamp,
            'sample_count': self.sample_count,
            'signature_hash': self.get_signature_hash()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LanguageCluster':
        """
        Create LanguageCluster from dictionary representation.
        
        Args:
            data: Dictionary containing cluster data
            
        Returns:
            New LanguageCluster instance
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ['cluster_id']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(
            cluster_id=data['cluster_id'],
            character_statistics=data.get('character_statistics', {}),
            confidence_threshold=data.get('confidence_threshold', 0.7),
            pattern_signatures=data.get('pattern_signatures', {}),
            creation_timestamp=data.get('creation_timestamp'),
            sample_count=data.get('sample_count', 0)
        )


class ScriptType:
    """
    Enumeration of script types for language detection.
    
    Represents different writing systems and their detection confidence.
    """
    
    def __init__(self, name: str, confidence: float, component_scripts: Optional[list] = None):
        """
        Initialize script type.
        
        Args:
            name: Name of the script type
            confidence: Detection confidence (0.0 to 1.0)
            component_scripts: List of component scripts for mixed types
        """
        self.name = name
        self.confidence = confidence
        self.component_scripts = component_scripts or []
    
    def __eq__(self, other):
        """Equality comparison based on name."""
        if isinstance(other, ScriptType):
            return self.name == other.name
        return False
    
    def __str__(self):
        """String representation."""
        return f"ScriptType({self.name}, confidence={self.confidence:.2f})"
    
    def __repr__(self):
        """Debug representation."""
        return self.__str__()


# Predefined script types
SCRIPT_TYPES = {
    'JAPANESE': ScriptType('JAPANESE', 1.0),
    'LATIN': ScriptType('LATIN', 1.0),  
    'MIXED': ScriptType('MIXED', 1.0),
    'UNKNOWN': ScriptType('UNKNOWN', 0.5),
    'CYRILLIC': ScriptType('CYRILLIC', 1.0),
    'ARABIC': ScriptType('ARABIC', 1.0),
    'DEVANAGARI': ScriptType('DEVANAGARI', 1.0),
    'CHINESE': ScriptType('CHINESE', 1.0)
}