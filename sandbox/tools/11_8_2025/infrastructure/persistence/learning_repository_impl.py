"""
Multilingual Learning Repository Implementation.

Concrete implementation of the multilingual learning repository using
file-based persistence. Follows the Repository pattern and implements
the abstract interface defined in the domain layer.
"""

import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib
import sqlite3
from dataclasses import asdict

from ...domain.repositories.multilingual_learning_repository import (
    MultilingualLearningRepository,
    RepositoryError,
    ClusterNotFoundError,
    SessionNotFoundError,
    PersistenceError
)
from ...domain.value_objects.language_cluster import LanguageCluster
from ...domain.entities.multilingual_tokenizer import MultilingualTokenizer
from ..config.system_config import SystemConfig


class LearningRepositoryImpl(MultilingualLearningRepository):
    """
    File-based implementation of multilingual learning repository.
    
    This implementation uses a combination of:
    - SQLite database for metadata and indexing
    - JSON files for human-readable cluster data
    - Pickle files for binary tokenizer state
    - Directory structure for organization
    
    Directory Structure:
    data/
    ├── sessions.db (SQLite database)
    ├── sessions/
    │   ├── {session_id}/
    │   │   ├── clusters/
    │   │   │   ├── {cluster_id}.json
    │   │   ├── checkpoints/
    │   │   │   ├── {checkpoint_name}.pkl
    │   │   ├── progress/
    │   │   │   ├── learning_history.jsonl
    │   │   └── metadata.json
    ├── corpus/
    │   ├── samples.db
    │   └── texts/
    │       ├── {sample_id}.txt
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize repository with configuration.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.data_dir = config.data_directory
        self.sessions_dir = self.data_dir / "sessions"
        self.corpus_dir = self.data_dir / "corpus"
        
        # Initialize directory structure
        self._ensure_directory_structure()
        
        # Initialize databases
        self.sessions_db_path = self.data_dir / "sessions.db"
        self.corpus_db_path = self.corpus_dir / "samples.db"
        
        # Initialize database connections
        asyncio.create_task(self._initialize_databases())
    
    async def save_language_cluster(
        self,
        cluster: LanguageCluster,
        session_id: str
    ) -> None:
        """Save a language cluster to persistent storage."""
        try:
            session_dir = self._get_session_dir(session_id)
            clusters_dir = session_dir / "clusters"
            clusters_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cluster as JSON
            cluster_file = clusters_dir / f"{cluster.cluster_id}.json"
            cluster_data = cluster.to_dict()
            
            with open(cluster_file, 'w', encoding='utf-8') as f:
                json.dump(cluster_data, f, indent=2, ensure_ascii=False)
            
            # Update database index
            await self._update_cluster_index(session_id, cluster)
            
        except Exception as e:
            raise PersistenceError(f"Failed to save cluster {cluster.cluster_id}: {e}")
    
    async def get_language_cluster(
        self,
        cluster_id: str,
        session_id: str
    ) -> Optional[LanguageCluster]:
        """Retrieve a specific language cluster."""
        try:
            cluster_file = self._get_session_dir(session_id) / "clusters" / f"{cluster_id}.json"
            
            if not cluster_file.exists():
                return None
            
            with open(cluster_file, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            
            return LanguageCluster.from_dict(cluster_data)
            
        except Exception as e:
            raise PersistenceError(f"Failed to load cluster {cluster_id}: {e}")
    
    async def get_all_language_clusters(
        self,
        session_id: str
    ) -> List[LanguageCluster]:
        """Retrieve all language clusters for a session."""
        try:
            clusters_dir = self._get_session_dir(session_id) / "clusters"
            
            if not clusters_dir.exists():
                return []
            
            clusters = []
            for cluster_file in clusters_dir.glob("*.json"):
                try:
                    with open(cluster_file, 'r', encoding='utf-8') as f:
                        cluster_data = json.load(f)
                    clusters.append(LanguageCluster.from_dict(cluster_data))
                except Exception as e:
                    # Log error but continue with other clusters
                    print(f"Warning: Failed to load cluster {cluster_file}: {e}")
            
            return clusters
            
        except Exception as e:
            raise PersistenceError(f"Failed to load clusters for session {session_id}: {e}")
    
    async def delete_language_cluster(
        self,
        cluster_id: str,
        session_id: str
    ) -> bool:
        """Delete a specific language cluster."""
        try:
            cluster_file = self._get_session_dir(session_id) / "clusters" / f"{cluster_id}.json"
            
            if not cluster_file.exists():
                return False
            
            cluster_file.unlink()
            
            # Remove from database index
            await self._remove_cluster_from_index(session_id, cluster_id)
            
            return True
            
        except Exception as e:
            raise PersistenceError(f"Failed to delete cluster {cluster_id}: {e}")
    
    async def save_tokenizer_state(
        self,
        tokenizer: MultilingualTokenizer,
        session_id: str,
        checkpoint_name: str
    ) -> None:
        """Save complete tokenizer state including all clusters."""
        try:
            session_dir = self._get_session_dir(session_id)
            checkpoints_dir = session_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tokenizer state as pickle for complete state preservation
            checkpoint_file = checkpoints_dir / f"{checkpoint_name}.pkl"
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(tokenizer, f)
            
            # Also save clusters individually for human readability
            for cluster_id, cluster in tokenizer.language_clusters.items():
                await self.save_language_cluster(cluster, session_id)
            
            # Update checkpoint metadata
            await self._update_checkpoint_metadata(session_id, checkpoint_name)
            
        except Exception as e:
            raise PersistenceError(f"Failed to save tokenizer checkpoint {checkpoint_name}: {e}")
    
    async def load_tokenizer_state(
        self,
        session_id: str,
        checkpoint_name: Optional[str] = None
    ) -> Optional[MultilingualTokenizer]:
        """Load tokenizer state from storage."""
        try:
            checkpoints_dir = self._get_session_dir(session_id) / "checkpoints"
            
            if not checkpoints_dir.exists():
                return None
            
            # Find checkpoint file
            if checkpoint_name is None:
                # Find latest checkpoint
                checkpoint_files = list(checkpoints_dir.glob("*.pkl"))
                if not checkpoint_files:
                    return None
                checkpoint_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            else:
                checkpoint_file = checkpoints_dir / f"{checkpoint_name}.pkl"
                if not checkpoint_file.exists():
                    return None
            
            # Load tokenizer state
            with open(checkpoint_file, 'rb') as f:
                tokenizer = pickle.load(f)
            
            return tokenizer
            
        except Exception as e:
            raise PersistenceError(f"Failed to load tokenizer checkpoint: {e}")
    
    async def save_learning_progress(
        self,
        session_id: str,
        text_sample: str,
        tokens: List[str],
        cluster_id: str,
        learning_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record learning progress for analysis and replay."""
        try:
            session_dir = self._get_session_dir(session_id)
            progress_dir = session_dir / "progress"
            progress_dir.mkdir(parents=True, exist_ok=True)
            
            # Create progress record
            progress_record = {
                'timestamp': (timestamp or datetime.now()).isoformat(),
                'text_sample_hash': hashlib.md5(text_sample.encode()).hexdigest(),
                'text_length': len(text_sample),
                'text_preview': text_sample[:200] + '...' if len(text_sample) > 200 else text_sample,
                'tokens': tokens,
                'token_count': len(tokens),
                'cluster_id': cluster_id,
                'learning_metrics': learning_metrics
            }
            
            # Append to learning history file (JSONL format)
            history_file = progress_dir / "learning_history.jsonl"
            with open(history_file, 'a', encoding='utf-8') as f:
                json.dump(progress_record, f, ensure_ascii=False)
                f.write('\n')
            
            # Update session statistics
            await self._update_session_statistics(session_id, progress_record)
            
        except Exception as e:
            raise PersistenceError(f"Failed to save learning progress: {e}")
    
    async def get_learning_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        cluster_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve learning history records."""
        try:
            history_file = self._get_session_dir(session_id) / "progress" / "learning_history.jsonl"
            
            if not history_file.exists():
                return []
            
            records = []
            with open(history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        
                        # Apply cluster filter if specified
                        if cluster_id and record.get('cluster_id') != cluster_id:
                            continue
                        
                        records.append(record)
                        
                        # Apply limit if specified
                        if limit and len(records) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
            
            return records
            
        except Exception as e:
            raise PersistenceError(f"Failed to retrieve learning history: {e}")
    
    async def save_corpus_sample(
        self,
        text: str,
        language_hint: Optional[str],
        source: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Store a text sample in the multilingual corpus."""
        try:
            # Generate sample ID
            sample_id = hashlib.sha256(
                f"{text}{source}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            # Save text file
            texts_dir = self.corpus_dir / "texts"
            texts_dir.mkdir(parents=True, exist_ok=True)
            
            text_file = texts_dir / f"{sample_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Save metadata to database
            await self._save_corpus_metadata(
                sample_id, text, language_hint, source, metadata
            )
            
            return sample_id
            
        except Exception as e:
            raise PersistenceError(f"Failed to save corpus sample: {e}")
    
    async def get_corpus_samples(
        self,
        language_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve corpus samples with optional filtering."""
        try:
            # Query from database with filters
            query = "SELECT * FROM corpus_samples WHERE 1=1"
            params = []
            
            if language_filter:
                query += " AND language_hint = ?"
                params.append(language_filter)
            
            if source_filter:
                query += " AND source = ?"
                params.append(source_filter)
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            async with self._get_corpus_db_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
            
            # Load text content for each sample
            samples = []
            for row in rows:
                sample_data = dict(row)
                
                # Load text content
                text_file = self.corpus_dir / "texts" / f"{sample_data['sample_id']}.txt"
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        sample_data['text_content'] = f.read()
                else:
                    sample_data['text_content'] = None
                
                samples.append(sample_data)
            
            return samples
            
        except Exception as e:
            raise PersistenceError(f"Failed to retrieve corpus samples: {e}")
    
    async def find_similar_clusters(
        self,
        reference_cluster: LanguageCluster,
        similarity_threshold: float = 0.8,
        session_id: Optional[str] = None
    ) -> List[Tuple[LanguageCluster, float]]:
        """Find clusters similar to the reference cluster."""
        try:
            similar_clusters = []
            
            if session_id:
                # Search within specific session
                sessions_to_search = [session_id]
            else:
                # Search across all sessions
                sessions_to_search = [d.name for d in self.sessions_dir.iterdir() if d.is_dir()]
            
            for search_session_id in sessions_to_search:
                try:
                    clusters = await self.get_all_language_clusters(search_session_id)
                    
                    for cluster in clusters:
                        similarity = reference_cluster.compute_similarity(cluster)
                        
                        if similarity >= similarity_threshold:
                            similar_clusters.append((cluster, similarity))
                            
                except Exception:
                    continue  # Skip sessions that can't be loaded
            
            # Sort by similarity (descending)
            similar_clusters.sort(key=lambda x: x[1], reverse=True)
            
            return similar_clusters
            
        except Exception as e:
            raise PersistenceError(f"Failed to find similar clusters: {e}")
    
    async def get_cluster_statistics(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive statistics about stored clusters."""
        try:
            stats = {
                'total_clusters': 0,
                'clusters_by_session': {},
                'average_sample_count': 0,
                'cluster_age_distribution': {},
                'total_sessions': 0
            }
            
            if session_id:
                sessions_to_analyze = [session_id]
            else:
                sessions_to_analyze = [d.name for d in self.sessions_dir.iterdir() if d.is_dir()]
            
            all_clusters = []
            
            for analyze_session_id in sessions_to_analyze:
                try:
                    clusters = await self.get_all_language_clusters(analyze_session_id)
                    all_clusters.extend(clusters)
                    
                    stats['clusters_by_session'][analyze_session_id] = len(clusters)
                    
                except Exception:
                    continue
            
            stats['total_clusters'] = len(all_clusters)
            stats['total_sessions'] = len(sessions_to_analyze)
            
            if all_clusters:
                stats['average_sample_count'] = sum(
                    cluster.sample_count for cluster in all_clusters
                ) / len(all_clusters)
                
                # Analyze cluster ages
                now = datetime.now().timestamp()
                for cluster in all_clusters:
                    if cluster.creation_timestamp:
                        age_days = (now - cluster.creation_timestamp) / (24 * 3600)
                        age_bucket = f"{int(age_days // 7)}w"  # Weekly buckets
                        stats['cluster_age_distribution'][age_bucket] = (
                            stats['cluster_age_distribution'].get(age_bucket, 0) + 1
                        )
            
            return stats
            
        except Exception as e:
            raise PersistenceError(f"Failed to get cluster statistics: {e}")
    
    async def cleanup_old_sessions(
        self,
        cutoff_date: datetime,
        preserve_checkpoints: bool = True
    ) -> int:
        """Clean up old session data to manage storage."""
        try:
            cleaned_sessions = 0
            cutoff_timestamp = cutoff_date.timestamp()
            
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                # Check session age
                session_timestamp = session_dir.stat().st_ctime
                
                if session_timestamp < cutoff_timestamp:
                    if preserve_checkpoints:
                        # Only remove progress and cluster data
                        progress_dir = session_dir / "progress"
                        clusters_dir = session_dir / "clusters"
                        
                        if progress_dir.exists():
                            import shutil
                            shutil.rmtree(progress_dir)
                        
                        if clusters_dir.exists():
                            shutil.rmtree(clusters_dir)
                    else:
                        # Remove entire session directory
                        import shutil
                        shutil.rmtree(session_dir)
                    
                    cleaned_sessions += 1
            
            return cleaned_sessions
            
        except Exception as e:
            raise PersistenceError(f"Failed to cleanup old sessions: {e}")
    
    async def export_session_data(
        self,
        session_id: str,
        export_format: str = "json"
    ) -> Dict[str, Any]:
        """Export complete session data for backup or analysis."""
        try:
            session_dir = self._get_session_dir(session_id)
            
            if not session_dir.exists():
                raise SessionNotFoundError(f"Session {session_id} not found")
            
            # Gather all session data
            export_data = {
                'session_id': session_id,
                'export_timestamp': datetime.now().isoformat(),
                'export_format': export_format
            }
            
            # Load session metadata
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    export_data['metadata'] = json.load(f)
            
            # Load all clusters
            clusters = await self.get_all_language_clusters(session_id)
            export_data['clusters'] = [cluster.to_dict() for cluster in clusters]
            
            # Load learning history
            learning_history = await self.get_learning_history(session_id)
            export_data['learning_history'] = learning_history
            
            # Include checkpoint information
            checkpoints_dir = session_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_files = [f.stem for f in checkpoints_dir.glob("*.pkl")]
                export_data['available_checkpoints'] = checkpoint_files
            
            return export_data
            
        except Exception as e:
            raise PersistenceError(f"Failed to export session {session_id}: {e}")
    
    async def import_session_data(
        self,
        session_data: Dict[str, Any],
        new_session_id: Optional[str] = None
    ) -> str:
        """Import session data from backup."""
        try:
            import uuid
            session_id = new_session_id or str(uuid.uuid4())
            
            session_dir = self._get_session_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Import metadata
            if 'metadata' in session_data:
                metadata_file = session_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(session_data['metadata'], f, indent=2)
            
            # Import clusters
            if 'clusters' in session_data:
                clusters_dir = session_dir / "clusters"
                clusters_dir.mkdir(exist_ok=True)
                
                for cluster_data in session_data['clusters']:
                    cluster = LanguageCluster.from_dict(cluster_data)
                    await self.save_language_cluster(cluster, session_id)
            
            # Import learning history
            if 'learning_history' in session_data:
                progress_dir = session_dir / "progress"
                progress_dir.mkdir(exist_ok=True)
                
                history_file = progress_dir / "learning_history.jsonl"
                with open(history_file, 'w', encoding='utf-8') as f:
                    for record in session_data['learning_history']:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')
            
            return session_id
            
        except Exception as e:
            raise PersistenceError(f"Failed to import session data: {e}")
    
    def _ensure_directory_structure(self) -> None:
        """Ensure required directory structure exists."""
        directories = [
            self.data_dir,
            self.sessions_dir,
            self.corpus_dir,
            self.corpus_dir / "texts"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_session_dir(self, session_id: str) -> Path:
        """Get directory path for a session."""
        return self.sessions_dir / session_id
    
    async def _initialize_databases(self) -> None:
        """Initialize SQLite databases."""
        try:
            # Initialize sessions database
            async with self._get_sessions_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT,
                        last_updated TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS clusters (
                        cluster_id TEXT,
                        session_id TEXT,
                        created_at TEXT,
                        sample_count INTEGER,
                        confidence_threshold REAL,
                        PRIMARY KEY (cluster_id, session_id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        session_id TEXT,
                        checkpoint_name TEXT,
                        created_at TEXT,
                        file_size INTEGER,
                        PRIMARY KEY (session_id, checkpoint_name)
                    )
                """)
            
            # Initialize corpus database
            async with self._get_corpus_db_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS corpus_samples (
                        sample_id TEXT PRIMARY KEY,
                        text_hash TEXT,
                        text_length INTEGER,
                        language_hint TEXT,
                        source TEXT,
                        created_at TEXT,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_language_hint 
                    ON corpus_samples(language_hint)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_source 
                    ON corpus_samples(source)
                """)
                
        except Exception as e:
            raise PersistenceError(f"Failed to initialize databases: {e}")
    
    async def _get_sessions_db_connection(self):
        """Get connection to sessions database."""
        # In a real implementation, this would use an async database library
        # For now, we'll use a simple synchronous connection with async wrapper
        return sqlite3.connect(self.sessions_db_path)
    
    async def _get_corpus_db_connection(self):
        """Get connection to corpus database."""
        # In a real implementation, this would use an async database library
        return sqlite3.connect(self.corpus_db_path)
    
    async def _update_cluster_index(
        self,
        session_id: str,
        cluster: LanguageCluster
    ) -> None:
        """Update cluster index in database."""
        try:
            async with self._get_sessions_db_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO clusters 
                    (cluster_id, session_id, created_at, sample_count, confidence_threshold)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    cluster.cluster_id,
                    session_id,
                    datetime.fromtimestamp(cluster.creation_timestamp or 0).isoformat(),
                    cluster.sample_count,
                    cluster.confidence_threshold
                ))
                conn.commit()
        except Exception as e:
            # Log error but don't fail the operation
            print(f"Warning: Failed to update cluster index: {e}")
    
    async def _remove_cluster_from_index(
        self,
        session_id: str,
        cluster_id: str
    ) -> None:
        """Remove cluster from database index."""
        try:
            async with self._get_sessions_db_connection() as conn:
                conn.execute(
                    "DELETE FROM clusters WHERE cluster_id = ? AND session_id = ?",
                    (cluster_id, session_id)
                )
                conn.commit()
        except Exception as e:
            print(f"Warning: Failed to remove cluster from index: {e}")
    
    async def _update_checkpoint_metadata(
        self,
        session_id: str,
        checkpoint_name: str
    ) -> None:
        """Update checkpoint metadata in database."""
        try:
            checkpoint_file = (
                self._get_session_dir(session_id) / "checkpoints" / f"{checkpoint_name}.pkl"
            )
            file_size = checkpoint_file.stat().st_size if checkpoint_file.exists() else 0
            
            async with self._get_sessions_db_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints 
                    (session_id, checkpoint_name, created_at, file_size)
                    VALUES (?, ?, ?, ?)
                """, (
                    session_id,
                    checkpoint_name,
                    datetime.now().isoformat(),
                    file_size
                ))
                conn.commit()
        except Exception as e:
            print(f"Warning: Failed to update checkpoint metadata: {e}")
    
    async def _save_corpus_metadata(
        self,
        sample_id: str,
        text: str,
        language_hint: Optional[str],
        source: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Save corpus sample metadata to database."""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            async with self._get_corpus_db_connection() as conn:
                conn.execute("""
                    INSERT INTO corpus_samples 
                    (sample_id, text_hash, text_length, language_hint, source, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample_id,
                    text_hash,
                    len(text),
                    language_hint,
                    source,
                    datetime.now().isoformat(),
                    json.dumps(metadata)
                ))
                conn.commit()
        except Exception as e:
            raise PersistenceError(f"Failed to save corpus metadata: {e}")
    
    async def _update_session_statistics(
        self,
        session_id: str,
        progress_record: Dict[str, Any]
    ) -> None:
        """Update session statistics based on learning progress."""
        try:
            session_dir = self._get_session_dir(session_id)
            metadata_file = session_dir / "metadata.json"
            
            # Load existing metadata
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    'session_id': session_id,
                    'created_at': datetime.now().isoformat(),
                    'statistics': {}
                }
            
            # Update statistics
            stats = metadata.setdefault('statistics', {})
            stats['total_learning_samples'] = stats.get('total_learning_samples', 0) + 1
            stats['total_tokens_generated'] = (
                stats.get('total_tokens_generated', 0) + progress_record['token_count']
            )
            stats['total_characters_processed'] = (
                stats.get('total_characters_processed', 0) + progress_record['text_length']
            )
            stats['last_learning_timestamp'] = progress_record['timestamp']
            
            metadata['last_updated'] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to update session statistics: {e}")