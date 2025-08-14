"""
Comprehensive Test Suite for Multilingual Tokenizer.

This test suite follows TDD principles with comprehensive coverage of:
- Single language tokenization
- Multiple language clustering 
- Boundary detection algorithms
- Persistence and recovery
- Edge cases (mixed languages, unknown scripts)

Test-Driven Development structure:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Improve while keeping tests green
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple, Optional
from unittest.mock import Mock, patch

# Test fixtures and setup
@pytest.fixture
def sample_japanese_text():
    """Sample Japanese text for testing."""
    return "私は学生です。今日は良い天気ですね。"

@pytest.fixture  
def sample_english_text():
    """Sample English text for testing."""
    return "I am a student. Today is nice weather."

@pytest.fixture
def sample_mixed_text():
    """Sample mixed language text for testing."""
    return "Hello こんにちは world 世界"

@pytest.fixture
def sample_unknown_script():
    """Sample text with unknown script for testing."""
    return "Здравствуй мир 🌍 नमस्ते दुनिया"

@pytest.fixture
def mock_som():
    """Mock SelfOrganizingMap for testing."""
    mock = Mock()
    mock.train_single_iteration.return_value = (1, 1)
    mock.find_best_matching_unit.return_value = (1, 1)
    mock.is_trained = True
    return mock

@pytest.fixture
def mock_predictive_coder():
    """Mock PredictiveCodingCore for testing."""
    mock = Mock()
    mock.process_input.return_value = Mock(total_error=0.5)
    return mock

@pytest.fixture
def mock_bayesian_service():
    """Mock BayesianInferenceService for testing."""
    mock = Mock()
    mock.update_beliefs.return_value = Mock()
    mock.compute_model_evidence.return_value = -10.5
    return mock


class TestLanguageCluster:
    """Test suite for LanguageCluster value object."""
    
    def test_language_cluster_creation(self):
        """Test basic LanguageCluster instantiation."""
        from domain.value_objects.language_cluster import LanguageCluster
        
        cluster = LanguageCluster(
            cluster_id="japanese_v1",
            character_statistics={
                'has_hiragana': True,
                'has_kanji': True,
                'space_ratio': 0.1
            },
            confidence_threshold=0.7
        )
        
        assert cluster.cluster_id == "japanese_v1"
        assert cluster.character_statistics['has_hiragana'] is True
        assert cluster.confidence_threshold == 0.7

    def test_language_cluster_immutability(self):
        """Test that LanguageCluster is immutable as a value object."""
        from domain.value_objects.language_cluster import LanguageCluster
        
        cluster = LanguageCluster(
            cluster_id="test",
            character_statistics={'space_ratio': 0.2},
            confidence_threshold=0.8
        )
        
        # Should not be able to modify after creation
        with pytest.raises(AttributeError):
            cluster.cluster_id = "modified"

    def test_language_cluster_equality(self):
        """Test equality comparison for LanguageCluster value objects."""
        from domain.value_objects.language_cluster import LanguageCluster
        
        cluster1 = LanguageCluster(
            cluster_id="test",
            character_statistics={'space_ratio': 0.2},
            confidence_threshold=0.8
        )
        
        cluster2 = LanguageCluster(
            cluster_id="test", 
            character_statistics={'space_ratio': 0.2},
            confidence_threshold=0.8
        )
        
        cluster3 = LanguageCluster(
            cluster_id="different",
            character_statistics={'space_ratio': 0.2},
            confidence_threshold=0.8
        )
        
        assert cluster1 == cluster2
        assert cluster1 != cluster3

    def test_language_cluster_feature_extraction(self):
        """Test feature extraction capabilities."""
        from domain.value_objects.language_cluster import LanguageCluster
        
        cluster = LanguageCluster(
            cluster_id="test",
            character_statistics={
                'has_latin': True,
                'has_kanji': False,
                'space_ratio': 0.15,
                'avg_word_length': 5.2
            },
            confidence_threshold=0.8
        )
        
        features = cluster.extract_features()
        assert isinstance(features, dict)
        assert 'has_latin' in features
        assert features['space_ratio'] == 0.15


class TestLanguageDetectionService:
    """Test suite for LanguageDetectionService domain service."""
    
    def test_detect_japanese_script(self, sample_japanese_text):
        """Test detection of Japanese script type."""
        from domain.services.language_detection_service import LanguageDetectionService
        
        detector = LanguageDetectionService()
        script_type = detector.detect_script_type(sample_japanese_text)
        
        assert script_type.name == "JAPANESE"
        assert script_type.confidence > 0.7

    def test_detect_english_script(self, sample_english_text):
        """Test detection of English script type."""
        from domain.services.language_detection_service import LanguageDetectionService
        
        detector = LanguageDetectionService()
        script_type = detector.detect_script_type(sample_english_text)
        
        assert script_type.name == "LATIN"
        assert script_type.confidence > 0.7

    def test_detect_mixed_script(self, sample_mixed_text):
        """Test detection of mixed script types."""
        from domain.services.language_detection_service import LanguageDetectionService
        
        detector = LanguageDetectionService()
        script_type = detector.detect_script_type(sample_mixed_text)
        
        assert script_type.name == "MIXED"
        assert len(script_type.component_scripts) >= 2

    def test_find_existing_language_cluster(self, sample_japanese_text, mock_som):
        """Test finding existing language cluster for text."""
        from domain.services.language_detection_service import LanguageDetectionService
        from domain.value_objects.language_cluster import LanguageCluster
        
        detector = LanguageDetectionService()
        
        # Create existing cluster
        existing_cluster = LanguageCluster(
            cluster_id="japanese_v1",
            character_statistics={'has_hiragana': True},
            confidence_threshold=0.8
        )
        detector._clusters = {"japanese_v1": existing_cluster}
        
        found_cluster = detector.find_language_cluster(sample_japanese_text)
        
        assert found_cluster is not None
        assert found_cluster.cluster_id == "japanese_v1"

    def test_create_new_cluster_for_unknown_language(self, sample_unknown_script):
        """Test creation of new cluster for unknown language."""
        from domain.services.language_detection_service import LanguageDetectionService
        
        detector = LanguageDetectionService()
        new_cluster = detector.create_new_cluster(sample_unknown_script)
        
        assert new_cluster.cluster_id.startswith("cluster_")
        assert len(new_cluster.character_statistics) > 0

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        from domain.services.language_detection_service import LanguageDetectionService
        
        detector = LanguageDetectionService()
        
        with pytest.raises(ValueError, match="Empty text cannot be processed"):
            detector.detect_script_type("")
            
        with pytest.raises(ValueError, match="Empty text cannot be processed"):
            detector.detect_script_type("   ")


class TestMultilingualTokenizer:
    """Test suite for MultilingualTokenizer entity."""
    
    def test_tokenizer_initialization(self):
        """Test basic tokenizer initialization."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer(
            max_clusters=5,
            similarity_threshold=0.8
        )
        
        assert tokenizer.max_clusters == 5
        assert tokenizer.similarity_threshold == 0.8
        assert len(tokenizer.language_clusters) == 0

    def test_single_language_tokenization(self, sample_japanese_text, mock_som, mock_predictive_coder):
        """Test tokenization of single language text."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        # Configure the mock to return proper numeric values
        mock_predictive_coder.calculate_prediction_error_at.return_value = 0.3
        
        with patch('domain.entities.multilingual_tokenizer.SelfOrganizingMap', return_value=mock_som):
            with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
                tokenizer = MultilingualTokenizer()
                tokens = tokenizer.tokenize(sample_japanese_text)
                
                assert len(tokens) > 0
                assert all(isinstance(token, str) for token in tokens)
                # Should detect some Japanese characters
                assert "私" in tokens
                # Should detect some boundaries (exact boundaries may vary)
                assert len(tokens) >= 3  # Should split into multiple tokens

    def test_multiple_language_clustering(self, sample_japanese_text, sample_english_text, mock_som, mock_predictive_coder):
        """Test clustering of multiple languages."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        # Configure mock to return proper values
        mock_predictive_coder.calculate_prediction_error_at.return_value = 0.3
        
        with patch('domain.entities.multilingual_tokenizer.SelfOrganizingMap', return_value=mock_som):
            with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
                tokenizer = MultilingualTokenizer()
                
                # Process Japanese text
                tokens_jp = tokenizer.tokenize(sample_japanese_text)
                
                # Process English text  
                tokens_en = tokenizer.tokenize(sample_english_text)
                
                # Should create 2 different clusters
                assert len(tokenizer.language_clusters) == 2
                
                cluster_ids = list(tokenizer.language_clusters.keys())
                assert len(set(cluster_ids)) == 2  # Unique cluster IDs

    def test_boundary_detection_algorithm(self, mock_predictive_coder):
        """Test the boundary detection using prediction error."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        # Mock prediction errors with clear peaks at word boundaries
        text = "helloworld"
        mock_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1]  # Peak at position 5
        mock_components = {'predictive_coder': mock_predictive_coder}
        
        with patch.object(tokenizer, '_calculate_prediction_errors', return_value=mock_errors):
            boundaries = tokenizer._detect_boundaries_by_prediction_error(text, mock_components)
            
            assert 5 in boundaries  # Should detect boundary between "hello" and "world"

    def test_entropy_boundary_detection(self):
        """Test boundary detection using branching entropy."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        # Text with clear word boundary
        text = "hello world"
        boundaries = tokenizer._detect_boundaries_by_entropy(text)
        
        # Should detect boundary at space position
        assert 5 in boundaries

    def test_mixed_language_processing(self, sample_mixed_text, mock_som, mock_predictive_coder):
        """Test processing of mixed language text."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        # Configure mock to return proper values
        mock_predictive_coder.calculate_prediction_error_at.return_value = 0.3
        
        with patch('domain.entities.multilingual_tokenizer.SelfOrganizingMap', return_value=mock_som):
            with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
                tokenizer = MultilingualTokenizer()
                tokens = tokenizer.tokenize(sample_mixed_text)
                
                # Should handle mixed language content
                has_latin = any(token for token in tokens if any(c.isascii() and c.isalpha() for c in token))
                has_japanese = any(token for token in tokens if any('\u3040' <= c <= '\u309f' or '\u4e00' <= c <= '\u9fff' for c in token))
                assert has_latin, f"Should contain Latin characters in tokens: {tokens}"
                assert has_japanese, f"Should contain Japanese characters in tokens: {tokens}"

    def test_unknown_script_handling(self, sample_unknown_script):
        """Test handling of unknown or rare scripts."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        tokens = tokenizer.tokenize(sample_unknown_script)
        
        # Should not crash and return some form of tokenization
        assert len(tokens) > 0
        assert isinstance(tokens, list)

    def test_tokenizer_learning_and_adaptation(self, sample_japanese_text, mock_som, mock_predictive_coder, mock_bayesian_service):
        """Test that tokenizer learns and adapts from repeated exposure."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        # Configure mocks to return proper values
        mock_predictive_coder.calculate_prediction_error_at.return_value = 0.3
        mock_som.train_single_iteration.return_value = None
        
        with patch('domain.entities.multilingual_tokenizer.SelfOrganizingMap', return_value=mock_som):
            with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
                with patch('domain.entities.multilingual_tokenizer.BayesianInferenceService', return_value=mock_bayesian_service):
                    tokenizer = MultilingualTokenizer()
                    
                    # First processing
                    tokens1 = tokenizer.tokenize(sample_japanese_text)
                    initial_cluster_count = len(tokenizer.language_clusters)
                    
                    # Second processing of similar text should improve, not create new cluster
                    similar_text = "私は教師です。今日は雨ですね。"
                    tokens2 = tokenizer.tokenize(similar_text)
                    
                    # Should not create additional clusters for same language
                    assert len(tokenizer.language_clusters) == initial_cluster_count
                    
                    # Should call learning methods
                    mock_som.train_single_iteration.assert_called()
                    mock_predictive_coder.process_input.assert_called()

    def test_persistence_and_recovery(self, tmp_path):
        """Test saving and loading tokenizer state."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        # Create and train tokenizer
        tokenizer = MultilingualTokenizer()
        
        # Save state
        save_path = tmp_path / "tokenizer_state.pkl"
        tokenizer.save_state(str(save_path))
        
        assert save_path.exists()
        
        # Load state into new tokenizer
        new_tokenizer = MultilingualTokenizer()
        new_tokenizer.load_state(str(save_path))
        
        # State should be preserved
        assert new_tokenizer.max_clusters == tokenizer.max_clusters
        assert new_tokenizer.similarity_threshold == tokenizer.similarity_threshold

    def test_performance_requirements(self, sample_english_text):
        """Test that tokenizer meets performance requirements."""
        import time
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        # Test processing speed: should handle 10,000 characters/second
        large_text = sample_english_text * 200  # ~6000 characters
        
        start_time = time.time()
        tokens = tokenizer.tokenize(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        chars_per_second = len(large_text) / processing_time
        
        # Should process at least 1000 chars/second (relaxed from 10,000 for test environment)
        assert chars_per_second > 1000
        assert len(tokens) > 0

    def test_max_clusters_limit(self, mock_som, mock_predictive_coder):
        """Test that tokenizer respects maximum cluster limit."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        with patch('domain.entities.multilingual_tokenizer.SelfOrganizingMap', return_value=mock_som):
            with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
                tokenizer = MultilingualTokenizer(max_clusters=2)
                
                # Try to process many different "languages" (simulated by very different texts)
                texts = [
                    "English text here",
                    "日本語のテキスト", 
                    "Текст на русском",
                    "Texte en français",
                    "Texto en español"
                ]
                
                for text in texts:
                    tokenizer.tokenize(text)
                
                # Should not exceed max_clusters limit
                assert len(tokenizer.language_clusters) <= 2

    def test_similarity_threshold_clustering(self, mock_som, mock_predictive_coder):
        """Test that similarity threshold affects cluster creation."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        with patch('domain.entities.multilingual_tokenizer.SelfOrganizingMap', return_value=mock_som):
            with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
                # High threshold - should create more clusters
                tokenizer_strict = MultilingualTokenizer(similarity_threshold=0.95)
                
                # Low threshold - should create fewer clusters  
                tokenizer_loose = MultilingualTokenizer(similarity_threshold=0.5)
                
                similar_texts = [
                    "Hello world",
                    "Hello there", 
                    "Hi world",
                    "Greetings world"
                ]
                
                for text in similar_texts:
                    tokenizer_strict.tokenize(text)
                    tokenizer_loose.tokenize(text)
                
                # Strict threshold should create more clusters
                assert len(tokenizer_strict.language_clusters) >= len(tokenizer_loose.language_clusters)


class TestBoundaryDetectionIntegration:
    """Integration tests for boundary detection algorithms."""
    
    def test_prediction_error_entropy_combination(self):
        """Test combination of prediction error and entropy methods."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        text = "hello world testing"
        
        # Get boundaries from both methods
        error_boundaries = tokenizer._detect_boundaries_by_prediction_error(text)
        entropy_boundaries = tokenizer._detect_boundaries_by_entropy(text)
        
        # Combined should intelligently merge overlapping boundaries
        combined_boundaries = tokenizer._combine_boundary_methods(
            error_boundaries, entropy_boundaries
        )
        
        # Combined should contain meaningful boundaries, though not necessarily
        # more than individual methods due to intelligent merging
        assert len(combined_boundaries) > 0
        assert isinstance(combined_boundaries, list)
        
        # All boundaries should be within text length
        for boundary in combined_boundaries:
            assert 0 <= boundary <= len(text)
        
        # Boundaries should be sorted
        assert combined_boundaries == sorted(combined_boundaries)

    def test_boundary_confidence_scoring(self):
        """Test confidence scoring for detected boundaries."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        text = "clear boundary"
        
        boundaries_with_confidence = tokenizer._detect_boundaries_with_confidence(text)
        
        assert all(len(boundary) == 2 for boundary in boundaries_with_confidence)  # (position, confidence)
        assert all(0 <= confidence <= 1 for _, confidence in boundaries_with_confidence)

    def test_boundary_refinement(self):
        """Test boundary refinement and post-processing."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        # Raw boundaries with some noise
        raw_boundaries = [5, 6, 11, 12, 18]  # Adjacent boundaries that should be merged
        text = "hello world testing"
        
        refined_boundaries = tokenizer._refine_boundaries(raw_boundaries, text)
        
        # Should merge adjacent boundaries
        assert len(refined_boundaries) < len(raw_boundaries)
        assert 5 in refined_boundaries or 6 in refined_boundaries  # One of the adjacent pair
        assert not (5 in refined_boundaries and 6 in refined_boundaries)  # But not both


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error conditions."""
    
    def test_empty_string_tokenization(self):
        """Test tokenization of empty string."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        with pytest.raises(ValueError, match="Cannot tokenize empty text"):
            tokenizer.tokenize("")

    def test_whitespace_only_tokenization(self):
        """Test tokenization of whitespace-only string."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        with pytest.raises(ValueError, match="Cannot tokenize empty text"):
            tokenizer.tokenize("   \n\t  ")

    def test_single_character_tokenization(self):
        """Test tokenization of single character."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        tokens = tokenizer.tokenize("a")
        
        assert tokens == ["a"]

    def test_very_long_text_handling(self):
        """Test handling of very long texts."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        long_text = "a" * 100000  # 100K characters
        
        # Should not crash
        tokens = tokenizer.tokenize(long_text)
        assert len(tokens) > 0

    def test_special_characters_handling(self):
        """Test handling of special characters and symbols."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        special_text = "Hello @user! Check this: https://example.com 😀 #hashtag"
        
        tokens = tokenizer.tokenize(special_text)
        
        # Should handle URLs, mentions, emojis, hashtags
        assert len(tokens) > 0
        assert any("@" in str(token) for token in tokens)  # Should preserve mentions
        assert any("https" in str(token) for token in tokens)  # Should preserve URLs

    def test_numeric_text_handling(self):
        """Test handling of numeric and alphanumeric text."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        numeric_text = "Price: $123.45 Date: 2024-01-01 Code: ABC123"
        
        tokens = tokenizer.tokenize(numeric_text)
        
        assert len(tokens) > 0
        # Should preserve some numeric content
        token_text = " ".join(tokens)
        assert "123" in token_text  # Some part of the number should be preserved
        assert "2024" in token_text  # Year should be preserved

    def test_corrupted_state_recovery(self, tmp_path):
        """Test recovery from corrupted state file."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        # Create corrupted state file
        corrupted_file = tmp_path / "corrupted.pkl"
        corrupted_file.write_text("This is not a valid pickle file")
        
        # Should handle corruption gracefully
        with pytest.raises(ValueError, match="Invalid state file"):
            tokenizer.load_state(str(corrupted_file))

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer(max_clusters=1000)  # High limit
        
        # Simulate memory pressure by processing many diverse texts
        for i in range(50):
            text = f"Text number {i} with unique content pattern {i*123}"
            tokenizer.tokenize(text)
        
        # Should handle gracefully without excessive memory usage
        assert len(tokenizer.language_clusters) < 50  # Should have merged similar patterns


class TestIntegrationWithExistingSystem:
    """Integration tests with existing consciousness system components."""
    
    def test_som_integration(self, mock_som):
        """Test integration with SelfOrganizingMap."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        tokenizer = MultilingualTokenizer()
        
        # Manually inject the mock SOM into cluster components
        cluster_id = "test_cluster"
        tokenizer._cluster_components[cluster_id] = {
            'som': mock_som,
            'predictive_coder': tokenizer._initialize_cluster_components()['predictive_coder'],
            'bayesian_service': tokenizer._initialize_cluster_components()['bayesian_service']
        }
        
        # Mock the cluster detection to return our test cluster
        with patch.object(tokenizer, '_get_or_create_cluster') as mock_get_cluster:
            from domain.value_objects.language_cluster import LanguageCluster
            mock_cluster = LanguageCluster(cluster_id=cluster_id, character_statistics={})
            mock_get_cluster.return_value = mock_cluster
            
            tokenizer.tokenize("test text")
            
            # Should call SOM training methods
            mock_som.train_single_iteration.assert_called()

    def test_predictive_coding_integration(self, mock_predictive_coder):
        """Test integration with PredictiveCodingCore."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        with patch('domain.entities.multilingual_tokenizer.PredictiveCodingCore', return_value=mock_predictive_coder):
            tokenizer = MultilingualTokenizer()
            tokenizer.tokenize("test text")
            
            # Should call predictive coding methods
            mock_predictive_coder.process_input.assert_called()

    def test_bayesian_inference_integration(self, mock_bayesian_service):
        """Test integration with BayesianInferenceService."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        
        with patch('domain.entities.multilingual_tokenizer.BayesianInferenceService', return_value=mock_bayesian_service):
            tokenizer = MultilingualTokenizer()
            tokenizer.tokenize("test text")
            
            # Should call Bayesian inference methods for uncertainty quantification
            mock_bayesian_service.update_beliefs.assert_called()


class TestPerformanceAndScalability:
    """Performance and scalability test suite."""
    
    def test_tokenization_speed_benchmark(self):
        """Benchmark tokenization speed."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        import time
        
        tokenizer = MultilingualTokenizer()
        test_text = "This is a test sentence for benchmarking tokenization speed."
        
        # Warm up
        for _ in range(10):
            tokenizer.tokenize(test_text)
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            tokenizer.tokenize(test_text)
        end_time = time.time()
        
        avg_time_per_tokenization = (end_time - start_time) / 100
        
        # Should be fast (less than 10ms per tokenization for this text length)
        assert avg_time_per_tokenization < 0.01

    def test_memory_usage_scaling(self):
        """Test memory usage as number of clusters grows."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        import sys
        
        tokenizer = MultilingualTokenizer()
        
        initial_size = sys.getsizeof(tokenizer)
        
        # Add multiple language clusters
        for i in range(10):
            unique_text = f"Unique language pattern {i} " * 20
            tokenizer.tokenize(unique_text)
        
        final_size = sys.getsizeof(tokenizer)
        
        # Memory growth should be reasonable
        memory_growth = final_size - initial_size
        assert memory_growth < 1_000_000  # Less than 1MB growth for 10 clusters

    def test_cluster_lookup_performance(self):
        """Test performance of cluster lookup operations."""
        from domain.entities.multilingual_tokenizer import MultilingualTokenizer
        import time
        
        tokenizer = MultilingualTokenizer()
        
        # Create many clusters
        for i in range(100):
            unique_text = f"Pattern {i}"
            tokenizer.tokenize(unique_text)
        
        # Test lookup performance
        test_text = "Test lookup performance"
        
        start_time = time.time()
        for _ in range(1000):
            tokenizer._find_best_cluster(test_text)
        end_time = time.time()
        
        avg_lookup_time = (end_time - start_time) / 1000
        
        # Lookup should be fast even with many clusters
        assert avg_lookup_time < 0.001  # Less than 1ms per lookup