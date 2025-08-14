"""
SentencePiece Integration Adapter.

Infrastructure adapter that integrates SentencePiece tokenization library
with the multilingual learning system. Provides a bridge between external
SentencePiece models and the domain's tokenization concepts.
"""

import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False


class SentencePieceAdapter:
    """
    Adapter for integrating SentencePiece with multilingual tokenizer.
    
    This adapter follows the Adapter pattern to integrate the external
    SentencePiece library with our domain model. It provides a clean
    interface that abstracts away SentencePiece specifics and translates
    between external library concepts and domain concepts.
    
    Integration Points:
    - Train SentencePiece models from text corpora
    - Use trained models for tokenization comparison
    - Extract vocabulary and statistics from models
    - Provide pre-tokenization for boundary hint generation
    """
    
    def __init__(self):
        """Initialize the SentencePiece adapter."""
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "SentencePiece library not available. Install with: pip install sentencepiece"
            )
        
        self._models: Dict[str, spm.SentencePieceProcessor] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
    
    def train_model_from_text(
        self,
        text_samples: List[str],
        model_name: str,
        vocab_size: int = 1000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995
    ) -> str:
        """
        Train a SentencePiece model from text samples.
        
        Args:
            text_samples: List of text samples for training
            model_name: Unique name for the model
            vocab_size: Target vocabulary size
            model_type: Model type ("bpe", "unigram", "char", "word")
            character_coverage: Character coverage for subword regularization
            
        Returns:
            Model identifier for future reference
            
        Raises:
            ValueError: If training fails or invalid parameters
        """
        if not text_samples:
            raise ValueError("Text samples cannot be empty")
        
        if model_name in self._models:
            raise ValueError(f"Model {model_name} already exists")
        
        try:
            # Create temporary files for training
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for text in text_samples:
                    f.write(text.strip() + '\n')
                input_file = f.name
            
            # Create temporary model files
            model_prefix = tempfile.mktemp()
            
            # Training parameters
            train_params = [
                f'--input={input_file}',
                f'--model_prefix={model_prefix}',
                f'--vocab_size={vocab_size}',
                f'--model_type={model_type}',
                f'--character_coverage={character_coverage}',
                '--user_defined_symbols=<pad>,<unk>,<s>,</s>',
                '--pad_id=0',
                '--unk_id=1',
                '--bos_id=2',
                '--eos_id=3',
                '--shuffle_input_sentence=true'
            ]
            
            # Train the model
            spm.SentencePieceTrainer.train(' '.join(train_params))
            
            # Load the trained model
            model = spm.SentencePieceProcessor()
            model.load(f'{model_prefix}.model')
            
            # Store model and metadata
            self._models[model_name] = model
            self._model_metadata[model_name] = {
                'vocab_size': vocab_size,
                'model_type': model_type,
                'character_coverage': character_coverage,
                'training_samples': len(text_samples),
                'model_file': f'{model_prefix}.model',
                'vocab_file': f'{model_prefix}.vocab'
            }
            
            # Clean up temporary input file
            Path(input_file).unlink()
            
            return model_name
            
        except Exception as e:
            raise ValueError(f"Failed to train SentencePiece model: {e}")
    
    def load_pretrained_model(
        self,
        model_path: str,
        model_name: str
    ) -> str:
        """
        Load a pretrained SentencePiece model.
        
        Args:
            model_path: Path to the .model file
            model_name: Unique name for the model
            
        Returns:
            Model identifier for future reference
        """
        if model_name in self._models:
            raise ValueError(f"Model {model_name} already exists")
        
        try:
            model = spm.SentencePieceProcessor()
            model.load(model_path)
            
            self._models[model_name] = model
            self._model_metadata[model_name] = {
                'model_file': model_path,
                'vocab_size': model.vocab_size(),
                'loaded_from': 'pretrained'
            }
            
            return model_name
            
        except Exception as e:
            raise ValueError(f"Failed to load pretrained model: {e}")
    
    def tokenize_with_model(
        self,
        text: str,
        model_name: str,
        enable_sampling: bool = False,
        alpha: float = 0.1,
        nbest_size: int = -1
    ) -> List[str]:
        """
        Tokenize text using a specific SentencePiece model.
        
        Args:
            text: Text to tokenize
            model_name: Name of the model to use
            enable_sampling: Enable subword regularization
            alpha: Sampling parameter for regularization
            nbest_size: Number of candidates for sampling
            
        Returns:
            List of token strings
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        
        try:
            if enable_sampling:
                tokens = model.sample_encode_as_pieces(
                    text,
                    nbest_size=nbest_size,
                    alpha=alpha
                )
            else:
                tokens = model.encode_as_pieces(text)
            
            return tokens
            
        except Exception as e:
            raise ValueError(f"Tokenization failed: {e}")
    
    def get_token_ids(
        self,
        text: str,
        model_name: str
    ) -> List[int]:
        """
        Get token IDs for text using a specific model.
        
        Args:
            text: Text to encode
            model_name: Name of the model to use
            
        Returns:
            List of token IDs
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        return model.encode_as_ids(text)
    
    def decode_tokens(
        self,
        tokens: List[str],
        model_name: str
    ) -> str:
        """
        Decode tokens back to text.
        
        Args:
            tokens: List of token strings
            model_name: Name of the model to use
            
        Returns:
            Decoded text string
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        return model.decode_pieces(tokens)
    
    def get_vocabulary(self, model_name: str) -> List[str]:
        """
        Get vocabulary from a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of vocabulary items
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        vocab = []
        
        for i in range(model.vocab_size()):
            vocab.append(model.id_to_piece(i))
        
        return vocab
    
    def get_token_scores(
        self,
        text: str,
        model_name: str
    ) -> List[Tuple[str, float]]:
        """
        Get tokens with their scores/probabilities.
        
        Args:
            text: Text to analyze
            model_name: Name of the model to use
            
        Returns:
            List of (token, score) tuples
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        
        try:
            # Encode with scores
            ids = model.encode_as_ids(text)
            pieces = model.encode_as_pieces(text)
            
            # Get scores for each token
            token_scores = []
            for piece_id, piece in zip(ids, pieces):
                # SentencePiece doesn't directly provide token probabilities
                # We use negative log-likelihood as a proxy for score
                score = -model.get_score(piece_id) if hasattr(model, 'get_score') else 1.0
                token_scores.append((piece, score))
            
            return token_scores
            
        except Exception as e:
            # Fallback to simple tokenization without scores
            pieces = model.encode_as_pieces(text)
            return [(piece, 1.0) for piece in pieces]
    
    def generate_boundary_hints(
        self,
        text: str,
        model_name: str,
        confidence_threshold: float = 0.5
    ) -> List[int]:
        """
        Generate boundary hints based on SentencePiece tokenization.
        
        This method provides boundary suggestions that can be used
        by the multilingual tokenizer as additional evidence for
        token boundary detection.
        
        Args:
            text: Text to analyze
            model_name: Name of the model to use
            confidence_threshold: Minimum confidence for boundary suggestions
            
        Returns:
            List of character positions where boundaries are suggested
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        
        try:
            # Get token pieces
            pieces = model.encode_as_pieces(text)
            
            # Calculate boundaries
            boundaries = []
            current_pos = 0
            
            for piece in pieces:
                # Remove the special prefix if present
                clean_piece = piece.replace('▁', ' ').lstrip()
                
                # Find the piece in the remaining text
                piece_start = text.find(clean_piece, current_pos)
                
                if piece_start >= 0:
                    # Add boundary at the start of each piece (except first)
                    if piece_start > current_pos:
                        boundaries.append(piece_start)
                    
                    current_pos = piece_start + len(clean_piece)
            
            return boundaries
            
        except Exception as e:
            # Return empty list if boundary generation fails
            return []
    
    def compare_tokenizations(
        self,
        text: str,
        custom_tokens: List[str],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Compare custom tokenization with SentencePiece tokenization.
        
        Args:
            text: Original text
            custom_tokens: Tokens from multilingual tokenizer
            model_name: SentencePiece model to compare against
            
        Returns:
            Comparison metrics and analysis
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get SentencePiece tokenization
        sp_tokens = self.tokenize_with_model(text, model_name)
        
        # Calculate comparison metrics
        custom_count = len(custom_tokens)
        sp_count = len(sp_tokens)
        
        # Token overlap analysis
        custom_set = set(custom_tokens)
        sp_set = set(sp_tokens)
        
        overlap = len(custom_set & sp_set)
        union_size = len(custom_set | sp_set)
        
        # Compression ratio comparison
        custom_compression = custom_count / len(text) if text else 0
        sp_compression = sp_count / len(text) if text else 0
        
        # Vocabulary efficiency
        custom_vocab_efficiency = len(custom_set) / custom_count if custom_count > 0 else 0
        sp_vocab_efficiency = len(sp_set) / sp_count if sp_count > 0 else 0
        
        return {
            'text_length': len(text),
            'custom_tokenization': {
                'token_count': custom_count,
                'unique_tokens': len(custom_set),
                'compression_ratio': custom_compression,
                'vocab_efficiency': custom_vocab_efficiency,
                'tokens': custom_tokens[:10]  # First 10 for display
            },
            'sentencepiece_tokenization': {
                'token_count': sp_count,
                'unique_tokens': len(sp_set),
                'compression_ratio': sp_compression,
                'vocab_efficiency': sp_vocab_efficiency,
                'tokens': sp_tokens[:10]  # First 10 for display
            },
            'comparison': {
                'token_overlap': overlap,
                'jaccard_similarity': overlap / union_size if union_size > 0 else 0,
                'compression_difference': abs(custom_compression - sp_compression),
                'efficiency_difference': abs(custom_vocab_efficiency - sp_vocab_efficiency),
                'custom_vs_sp_ratio': custom_count / sp_count if sp_count > 0 else float('inf')
            }
        }
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self._models[model_name]
        metadata = self._model_metadata[model_name]
        
        return {
            'model_name': model_name,
            'vocab_size': model.vocab_size(),
            'metadata': metadata,
            'is_loaded': True,
            'special_tokens': {
                'pad_id': model.pad_id(),
                'unk_id': model.unk_id(),
                'bos_id': model.bos_id(),
                'eos_id': model.eos_id()
            }
        }
    
    def list_models(self) -> List[str]:
        """
        List all loaded model names.
        
        Returns:
            List of model names
        """
        return list(self._models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        if model_name in self._models:
            del self._models[model_name]
            del self._model_metadata[model_name]
            return True
        return False
    
    def save_model(
        self,
        model_name: str,
        output_path: str
    ) -> bool:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            output_path: Path where to save the model
            
        Returns:
            True if model was saved successfully
        """
        if model_name not in self._models:
            return False
        
        try:
            metadata = self._model_metadata[model_name]
            model_file = metadata.get('model_file')
            
            if model_file and Path(model_file).exists():
                # Copy model file to output path
                import shutil
                shutil.copy2(model_file, output_path)
                return True
            
            return False
            
        except Exception:
            return False
    
    def analyze_text_characteristics(
        self,
        text: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Analyze text characteristics using SentencePiece model.
        
        Args:
            text: Text to analyze
            model_name: Model to use for analysis
            
        Returns:
            Analysis results including statistics and patterns
        """
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get tokenization
        tokens = self.tokenize_with_model(text, model_name)
        token_ids = self.get_token_ids(text, model_name)
        
        # Analyze token statistics
        token_lengths = [len(token.replace('▁', '')) for token in tokens]
        
        # Character-level analysis
        total_chars = len(text)
        subword_chars = sum(len(token.replace('▁', '')) for token in tokens)
        
        # OOV analysis (tokens that are likely out-of-vocabulary)
        model = self._models[model_name]
        unk_id = model.unk_id()
        oov_count = sum(1 for token_id in token_ids if token_id == unk_id)
        
        return {
            'text_length': total_chars,
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'compression_ratio': len(tokens) / total_chars if total_chars > 0 else 0,
            'oov_rate': oov_count / len(tokens) if tokens else 0,
            'avg_token_length': np.mean(token_lengths) if token_lengths else 0,
            'token_length_std': np.std(token_lengths) if token_lengths else 0,
            'character_recovery_rate': subword_chars / total_chars if total_chars > 0 else 0,
            'token_statistics': {
                'min_length': min(token_lengths) if token_lengths else 0,
                'max_length': max(token_lengths) if token_lengths else 0,
                'median_length': np.median(token_lengths) if token_lengths else 0
            }
        }