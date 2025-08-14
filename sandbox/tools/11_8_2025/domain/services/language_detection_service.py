"""
Language Detection Domain Service.

Domain service responsible for automatic language detection and cluster
management in the multilingual system. Implements statistical pattern
recognition and clustering algorithms for language identification.
"""

import re
import time
import uuid
from typing import Dict, Optional, List, Any
import numpy as np
from ..value_objects.language_cluster import LanguageCluster, ScriptType, SCRIPT_TYPES


class LanguageClusterFactory:
    """
    Factory for creating language clusters.
    
    Implements the Factory Method pattern to encapsulate cluster creation
    logic and provide different cluster creation strategies based on text
    characteristics and system configuration.
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def create_cluster_from_text(
        self, 
        text: str, 
        cluster_id: Optional[str] = None
    ) -> LanguageCluster:
        """
        Create a language cluster from text sample.
        
        Args:
            text: Representative text for the cluster
            cluster_id: Optional cluster ID (generated if not provided)
            
        Returns:
            New LanguageCluster instance
        """
        if cluster_id is None:
            cluster_id = self._generate_cluster_id(text)
        
        features = self._extract_cluster_features(text)
        
        return LanguageCluster(
            cluster_id=cluster_id,
            character_statistics=features['character_statistics'],
            confidence_threshold=self.similarity_threshold,
            pattern_signatures=features['pattern_signatures'],
            creation_timestamp=time.time(),
            sample_count=1
        )
    
    def create_minimal_cluster(self, cluster_id: str) -> LanguageCluster:
        """Create minimal cluster for testing purposes."""
        return LanguageCluster(
            cluster_id=cluster_id,
            character_statistics={},
            confidence_threshold=self.similarity_threshold,
            pattern_signatures={},
            creation_timestamp=time.time(),
            sample_count=0
        )
    
    def _generate_cluster_id(self, text: str) -> str:
        """Generate unique cluster ID based on text characteristics."""
        import uuid
        text_hash = hash(text) % 10000  # Simple hash for uniqueness
        return f"cluster_{text_hash}_{uuid.uuid4().hex[:8]}"
    
    def _extract_cluster_features(self, text: str) -> Dict[str, Any]:
        """Extract features for cluster creation."""
        # Extract more comprehensive features for better clustering
        character_stats = self._extract_character_statistics(text)
        pattern_sigs = self._extract_pattern_signatures(text)
        
        return {
            'character_statistics': character_stats,
            'pattern_signatures': pattern_sigs
        }
    
    def _extract_character_statistics(self, text: str) -> Dict[str, Any]:
        """Extract character-based statistical features."""
        if not text:
            return {}
        
        stats = {
            'text_length': len(text),
            'char_diversity': len(set(text)) / len(text),
            'space_ratio': text.count(' ') / len(text),
        }
        
        # Script detection features
        stats['has_latin'] = bool(re.search(r'[a-zA-Z]', text))
        stats['has_kanji'] = bool(re.search(r'[\u4e00-\u9fff]', text))
        stats['has_hiragana'] = bool(re.search(r'[\u3040-\u309f]', text))
        stats['has_katakana'] = bool(re.search(r'[\u30a0-\u30ff]', text))
        stats['has_cyrillic'] = bool(re.search(r'[\u0400-\u04ff]', text))
        stats['has_arabic'] = bool(re.search(r'[\u0600-\u06ff]', text))
        
        # Calculate script ratios
        total_chars = len(text)
        if total_chars > 0:
            stats['latin_ratio'] = len(re.findall(r'[a-zA-Z]', text)) / total_chars
            stats['kanji_ratio'] = len(re.findall(r'[\u4e00-\u9fff]', text)) / total_chars
            stats['hiragana_ratio'] = len(re.findall(r'[\u3040-\u309f]', text)) / total_chars
            stats['katakana_ratio'] = len(re.findall(r'[\u30a0-\u30ff]', text)) / total_chars
            
            # Composite Japanese ratio
            stats['japanese_ratio'] = (
                stats['kanji_ratio'] + 
                stats['hiragana_ratio'] + 
                stats['katakana_ratio']
            )
        
        return stats
    
    def _extract_pattern_signatures(self, text: str) -> Dict[str, float]:
        """Extract pattern signature features."""
        if not text:
            return {}
        
        return {
            'entropy': self._calculate_simple_entropy(text),
            'bigram_frequency': self._calculate_bigram_frequency(text),
            'char_distribution_skew': self._calculate_char_distribution_skew(text)
        }
    
    def _calculate_bigram_frequency(self, text: str) -> float:
        """Calculate average bigram frequency."""
        if len(text) < 2:
            return 0.0
        
        bigram_counts = {}
        total_bigrams = 0
        
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            total_bigrams += 1
        
        if total_bigrams == 0:
            return 0.0
        
        frequencies = list(bigram_counts.values())
        return sum(frequencies) / (len(frequencies) * total_bigrams)
    
    def _calculate_char_distribution_skew(self, text: str) -> float:
        """Calculate skewness of character distribution."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if len(char_counts) < 2:
            return 0.0
        
        frequencies = list(char_counts.values())
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        if std_freq == 0:
            return 0.0
        
        skewness = np.mean([((freq - mean_freq) / std_freq) ** 3 for freq in frequencies])
        return float(skewness)
    
    def _calculate_simple_entropy(self, text: str) -> float:
        """Calculate simple entropy for pattern signature."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy


class LanguageDetectionService:
    """
    Domain service for language detection and cluster management.
    
    This service encapsulates complex language detection logic that spans
    multiple domain objects and doesn't naturally belong to any single entity.
    
    Responsibilities:
    - Script type detection from text characteristics
    - Language cluster discovery and management
    - Feature extraction for clustering
    - Similarity computation between text and clusters
    """
    
    def __init__(self, max_clusters: int = 20, similarity_threshold: float = 0.8):
        """
        Initialize the language detection service.
        
        Args:
            max_clusters: Maximum number of language clusters to maintain
            similarity_threshold: Threshold for cluster similarity matching
        """
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold
        self._clusters: Dict[str, LanguageCluster] = {}
        self._feature_extractors = self._initialize_feature_extractors()
        self._cluster_factory = LanguageClusterFactory(similarity_threshold)
    
    def detect_script_type(self, text: str) -> ScriptType:
        """
        Detect the script type of input text.
        
        Analyzes character patterns to identify the writing system(s)
        used in the text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ScriptType object with detection results
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            raise ValueError("Empty text cannot be processed")
        
        # Extract character-based features
        features = self._extract_character_features(text)
        
        # Detect script types based on character patterns
        detected_scripts = []
        
        if features['has_hiragana'] or features['has_katakana'] or features['has_kanji']:
            detected_scripts.append(('JAPANESE', features['japanese_ratio']))
        
        if features['has_latin']:
            detected_scripts.append(('LATIN', features['latin_ratio']))
        
        if features['has_cyrillic']:
            detected_scripts.append(('CYRILLIC', features['cyrillic_ratio']))
        
        if features['has_arabic']:
            detected_scripts.append(('ARABIC', features['arabic_ratio']))
        
        if features['has_devanagari']:
            detected_scripts.append(('DEVANAGARI', features['devanagari_ratio']))
        
        # Determine primary script
        if not detected_scripts:
            return SCRIPT_TYPES['UNKNOWN']
        
        if len(detected_scripts) == 1:
            script_name, confidence = detected_scripts[0]
            return ScriptType(script_name, confidence)
        
        # Multiple scripts detected
        script_names = [script for script, _ in detected_scripts]
        return ScriptType('MIXED', 0.9, component_scripts=script_names)
    
    def find_language_cluster(self, text: str) -> Optional[LanguageCluster]:
        """
        Find the best matching language cluster for the given text.
        
        Args:
            text: Text to match against existing clusters
            
        Returns:
            Best matching LanguageCluster or None if no good match found
        """
        if not self._clusters:
            return None
        
        text_features = self._extract_all_features(text)
        best_cluster = None
        best_similarity = 0.0
        
        for cluster in self._clusters.values():
            similarity = self._compute_text_cluster_similarity(text_features, cluster)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        # Return cluster only if similarity exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return best_cluster
        
        return None
    
    def create_new_cluster(self, text: str) -> LanguageCluster:
        """
        Create a new language cluster from the given text.
        
        Args:
            text: Representative text for the new cluster
            
        Returns:
            New LanguageCluster instance
        """
        # Use factory to create cluster with proper feature extraction
        cluster = self._cluster_factory.create_cluster_from_text(text)
        
        # Add to cluster registry
        self._clusters[cluster.cluster_id] = cluster
        
        # Enforce cluster limit
        self._enforce_cluster_limit()
        
        return cluster
    
    def update_cluster_with_text(self, cluster_id: str, text: str) -> LanguageCluster:
        """
        Update an existing cluster with new text sample.
        
        Args:
            cluster_id: ID of cluster to update
            text: New text sample to incorporate
            
        Returns:
            Updated LanguageCluster instance
            
        Raises:
            ValueError: If cluster_id doesn't exist
        """
        if cluster_id not in self._clusters:
            raise ValueError(f"Cluster {cluster_id} not found")
        
        current_cluster = self._clusters[cluster_id]
        new_features = self._extract_all_features(text)
        
        # Merge features with existing statistics
        updated_statistics = self._merge_statistics(
            current_cluster.character_statistics,
            new_features['character_statistics'],
            current_cluster.sample_count
        )
        
        updated_patterns = self._merge_patterns(
            current_cluster.pattern_signatures,
            new_features['pattern_signatures'],
            current_cluster.sample_count
        )
        
        # Create updated cluster
        updated_cluster = current_cluster.with_updated_statistics(updated_statistics)
        updated_cluster = updated_cluster.with_updated_confidence(
            min(1.0, current_cluster.confidence_threshold + 0.01)  # Slight confidence boost
        )
        
        # Update sample count
        from dataclasses import replace
        updated_cluster = replace(
            updated_cluster,
            pattern_signatures=updated_patterns,
            sample_count=current_cluster.sample_count + 1
        )
        
        # Update registry
        self._clusters[cluster_id] = updated_cluster
        
        return updated_cluster
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about current clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        return {
            'total_clusters': len(self._clusters),
            'max_clusters': self.max_clusters,
            'similarity_threshold': self.similarity_threshold,
            'cluster_ids': list(self._clusters.keys()),
            'cluster_sizes': {
                cluster_id: cluster.sample_count
                for cluster_id, cluster in self._clusters.items()
            }
        }
    
    def _extract_character_features(self, text: str) -> Dict[str, Any]:
        """Extract character-based features from text."""
        features = {}
        
        # Unicode script detection
        features['has_latin'] = bool(re.search(r'[a-zA-Z]', text))
        features['has_kanji'] = bool(re.search(r'[\u4e00-\u9fff]', text))
        features['has_hiragana'] = bool(re.search(r'[\u3040-\u309f]', text))
        features['has_katakana'] = bool(re.search(r'[\u30a0-\u30ff]', text))
        features['has_cyrillic'] = bool(re.search(r'[\u0400-\u04ff]', text))
        features['has_arabic'] = bool(re.search(r'[\u0600-\u06ff]', text))
        features['has_devanagari'] = bool(re.search(r'[\u0900-\u097f]', text))
        
        # Character ratios
        total_chars = len(text)
        if total_chars > 0:
            features['latin_ratio'] = len(re.findall(r'[a-zA-Z]', text)) / total_chars
            features['kanji_ratio'] = len(re.findall(r'[\u4e00-\u9fff]', text)) / total_chars
            features['hiragana_ratio'] = len(re.findall(r'[\u3040-\u309f]', text)) / total_chars
            features['katakana_ratio'] = len(re.findall(r'[\u30a0-\u30ff]', text)) / total_chars
            features['cyrillic_ratio'] = len(re.findall(r'[\u0400-\u04ff]', text)) / total_chars
            features['arabic_ratio'] = len(re.findall(r'[\u0600-\u06ff]', text)) / total_chars
            features['devanagari_ratio'] = len(re.findall(r'[\u0900-\u097f]', text)) / total_chars
            
            # Composite ratios
            features['japanese_ratio'] = (
                features['kanji_ratio'] + 
                features['hiragana_ratio'] + 
                features['katakana_ratio']
            )
        else:
            # Default ratios for empty text
            for ratio_key in ['latin_ratio', 'kanji_ratio', 'hiragana_ratio', 
                             'katakana_ratio', 'cyrillic_ratio', 'arabic_ratio', 
                             'devanagari_ratio', 'japanese_ratio']:
                features[ratio_key] = 0.0
        
        # Statistical features
        features['space_ratio'] = text.count(' ') / len(text) if text else 0.0
        features['digit_ratio'] = len(re.findall(r'\d', text)) / len(text) if text else 0.0
        features['punctuation_ratio'] = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0.0
        features['char_diversity'] = len(set(text)) / len(text) if text else 0.0
        features['avg_word_length'] = self._estimate_avg_word_length(text)
        
        return features
    
    def _extract_all_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features for clustering."""
        character_stats = self._extract_character_features(text)
        
        # Pattern signatures (simplified statistical patterns)
        pattern_sigs = {
            'entropy': self._calculate_text_entropy(text),
            'bigram_frequency': self._calculate_bigram_frequency(text),
            'char_distribution_skew': self._calculate_char_distribution_skew(text)
        }
        
        return {
            'character_statistics': character_stats,
            'pattern_signatures': pattern_sigs
        }
    
    def _compute_text_cluster_similarity(
        self, 
        text_features: Dict[str, Any], 
        cluster: LanguageCluster
    ) -> float:
        """Compute similarity between text features and cluster."""
        # Use language-aware similarity calculation instead of generic cluster similarity
        return self._calculate_language_aware_similarity(text_features, cluster)
    
    def _calculate_language_aware_similarity(
        self,
        text_features: Dict[str, Any],
        cluster: LanguageCluster
    ) -> float:
        """Calculate similarity with language-specific weighting."""
        char_stats_sim = self._compare_character_statistics(
            text_features['character_statistics'], 
            cluster.character_statistics
        )
        
        pattern_sim = self._compare_pattern_signatures(
            text_features['pattern_signatures'],
            cluster.pattern_signatures
        )
        
        # Weight character statistics more heavily for language detection
        return 0.7 * char_stats_sim + 0.3 * pattern_sim
    
    def _compare_character_statistics(
        self, 
        stats1: Dict[str, Any], 
        stats2: Dict[str, Any]
    ) -> float:
        """Compare character statistics with language focus."""
        if not stats1 or not stats2:
            return 0.0
        
        # Focus on script-specific features for language detection
        script_features = [
            'has_latin', 'has_kanji', 'has_hiragana', 'has_katakana',
            'has_cyrillic', 'has_arabic'
        ]
        
        # Check script type similarity
        script_similarity = 0.0
        script_count = 0
        
        for feature in script_features:
            if feature in stats1 and feature in stats2:
                if stats1[feature] == stats2[feature]:
                    script_similarity += 1.0
                script_count += 1
        
        if script_count > 0:
            script_similarity /= script_count
        
        # For same script type, check ratios
        ratio_similarity = 0.0
        if script_similarity > 0.8:  # Same script family
            ratio_features = ['japanese_ratio', 'latin_ratio']
            ratio_count = 0
            
            for feature in ratio_features:
                if feature in stats1 and feature in stats2:
                    val1, val2 = stats1[feature], stats2[feature]
                    if val1 > 0 or val2 > 0:  # At least one has this script
                        max_val = max(val1, val2, 0.1)
                        diff = abs(val1 - val2) / max_val
                        ratio_similarity += max(0.0, 1.0 - diff)
                        ratio_count += 1
            
            if ratio_count > 0:
                ratio_similarity /= ratio_count
        
        # Combine script and ratio similarities
        return 0.8 * script_similarity + 0.2 * ratio_similarity
    
    def _compare_pattern_signatures(
        self,
        patterns1: Dict[str, float],
        patterns2: Dict[str, float]
    ) -> float:
        """Compare pattern signatures."""
        if not patterns1 or not patterns2:
            return 0.0
        
        similarities = []
        for key in patterns1:
            if key in patterns2:
                val1, val2 = patterns1[key], patterns2[key]
                max_val = max(abs(val1), abs(val2), 1e-8)
                diff = abs(val1 - val2) / max_val
                similarities.append(max(0.0, 1.0 - diff))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _merge_statistics(
        self, 
        current_stats: Dict[str, Any], 
        new_stats: Dict[str, Any], 
        current_count: int
    ) -> Dict[str, Any]:
        """Merge new statistics with existing ones using weighted average."""
        merged = {}
        total_count = current_count + 1
        
        for key in set(current_stats.keys()) | set(new_stats.keys()):
            current_val = current_stats.get(key, 0.0)
            new_val = new_stats.get(key, 0.0)
            
            if isinstance(current_val, bool) and isinstance(new_val, bool):
                # For boolean values, use OR logic
                merged[key] = current_val or new_val
            elif isinstance(current_val, (int, float)) and isinstance(new_val, (int, float)):
                # For numeric values, use weighted average
                merged[key] = (current_val * current_count + new_val) / total_count
            else:
                # For other types, prefer new value
                merged[key] = new_val
        
        return merged
    
    def _merge_patterns(
        self, 
        current_patterns: Dict[str, float], 
        new_patterns: Dict[str, float], 
        current_count: int
    ) -> Dict[str, float]:
        """Merge pattern signatures using weighted average."""
        merged = {}
        total_count = current_count + 1
        
        for key in set(current_patterns.keys()) | set(new_patterns.keys()):
            current_val = current_patterns.get(key, 0.0)
            new_val = new_patterns.get(key, 0.0)
            merged[key] = (current_val * current_count + new_val) / total_count
        
        return merged
    
    def _enforce_cluster_limit(self):
        """Enforce maximum cluster limit by removing least used clusters."""
        if len(self._clusters) <= self.max_clusters:
            return
        
        # Sort clusters by sample count (ascending)
        sorted_clusters = sorted(
            self._clusters.items(),
            key=lambda x: x[1].sample_count
        )
        
        # Remove clusters with lowest sample counts
        clusters_to_remove = len(self._clusters) - self.max_clusters
        for i in range(clusters_to_remove):
            cluster_id = sorted_clusters[i][0]
            del self._clusters[cluster_id]
    
    def _initialize_feature_extractors(self) -> Dict[str, Any]:
        """Initialize feature extraction components."""
        return {
            'character_patterns': re.compile(r'[\w\s\d]'),
            'word_boundaries': re.compile(r'\b'),
            'punctuation': re.compile(r'[^\w\s]')
        }
    
    def _estimate_avg_word_length(self, text: str) -> float:
        """Estimate average word length in text."""
        if not text.strip():
            return 0.0
        
        # Split on whitespace and filter empty strings
        words = [word for word in text.split() if word.strip()]
        
        if not words:
            return 0.0
        
        total_length = sum(len(word) for word in words)
        return total_length / len(words)
    
    def _calculate_text_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of character distribution."""
        if not text:
            return 0.0
        
        # Calculate character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _calculate_bigram_frequency(self, text: str) -> float:
        """Calculate average bigram frequency as a pattern signature."""
        if len(text) < 2:
            return 0.0
        
        bigram_counts = {}
        total_bigrams = 0
        
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            total_bigrams += 1
        
        if total_bigrams == 0:
            return 0.0
        
        # Return normalized average frequency
        frequencies = list(bigram_counts.values())
        return sum(frequencies) / (len(frequencies) * total_bigrams)
    
    def _calculate_char_distribution_skew(self, text: str) -> float:
        """Calculate skewness of character distribution."""
        if not text:
            return 0.0
        
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if len(char_counts) < 2:
            return 0.0
        
        frequencies = list(char_counts.values())
        mean_freq = np.mean(frequencies)
        std_freq = np.std(frequencies)
        
        if std_freq == 0:
            return 0.0
        
        # Calculate skewness
        skewness = np.mean([((freq - mean_freq) / std_freq) ** 3 for freq in frequencies])
        return float(skewness)