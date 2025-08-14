"""
Multilingual Learning Command Line Interface.

CLI interface for multilingual learning operations providing a user-friendly
command-line interface for all system functionality following Clean Architecture.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from ...application.services.multilingual_learning_service import MultilingualLearningService
from ...infrastructure.persistence.learning_repository_impl import LearningRepositoryImpl
from ...infrastructure.external.sentencepiece_adapter import SentencePieceAdapter
from ...infrastructure.config.system_config import SystemConfig, LogLevel


class MultilingualCLI:
    """
    Command Line Interface for multilingual learning system.
    
    This CLI provides a user-friendly interface for all multilingual learning
    operations while maintaining clean separation from business logic.
    Follows the Adapter pattern to bridge between CLI commands and application services.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize CLI with dependencies.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize infrastructure
        self.repository = LearningRepositoryImpl(config)
        
        # Initialize application service
        self.learning_service = MultilingualLearningService(self.repository, config)
        
        # Initialize external adapters
        try:
            self.sentencepiece_adapter = SentencePieceAdapter()
        except ImportError:
            self.sentencepiece_adapter = None
        
        # CLI state
        self.current_session_id: Optional[str] = None
        self.output_format = "json"
        self.verbose = False
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for CLI commands."""
        parser = argparse.ArgumentParser(
            description="Multilingual Learning System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Create a new learning session
  python -m multilingual_cli session create --name "Japanese Study Session"
  
  # Learn from text
  python -m multilingual_cli learn --text "こんにちは世界" --session-id abc123
  
  # Tokenize text
  python -m multilingual_cli tokenize --text "Hello world" --session-id abc123
  
  # Load from file
  python -m multilingual_cli learn --file corpus.txt --session-id abc123
  
  # Export session
  python -m multilingual_cli session export --session-id abc123 --output session.json
            """
        )
        
        # Global options
        parser.add_argument(
            '--config', 
            type=str, 
            help='Path to configuration file'
        )
        parser.add_argument(
            '--verbose', '-v', 
            action='store_true', 
            help='Enable verbose output'
        )
        parser.add_argument(
            '--output-format', 
            choices=['json', 'yaml', 'table'], 
            default='json',
            help='Output format (default: json)'
        )
        parser.add_argument(
            '--session-id', 
            type=str, 
            help='Session ID to use for operations'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Session management commands
        self._add_session_commands(subparsers)
        
        # Learning commands
        self._add_learning_commands(subparsers)
        
        # Tokenization commands
        self._add_tokenization_commands(subparsers)
        
        # Analysis commands
        self._add_analysis_commands(subparsers)
        
        # Comparison commands
        self._add_comparison_commands(subparsers)
        
        # Utility commands
        self._add_utility_commands(subparsers)
        
        return parser
    
    def _add_session_commands(self, subparsers):
        """Add session management commands."""
        session_parser = subparsers.add_parser('session', help='Session management')
        session_subparsers = session_parser.add_subparsers(dest='session_command')
        
        # Create session
        create_parser = session_subparsers.add_parser('create', help='Create new session')
        create_parser.add_argument('--name', type=str, help='Session name')
        create_parser.add_argument('--max-clusters', type=int, default=20, help='Maximum clusters')
        create_parser.add_argument('--similarity-threshold', type=float, default=0.8, help='Similarity threshold')
        create_parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
        
        # List sessions
        session_subparsers.add_parser('list', help='List active sessions')
        
        # Show session info
        info_parser = session_subparsers.add_parser('info', help='Show session information')
        info_parser.add_argument('session_id', help='Session ID')
        
        # Export session
        export_parser = session_subparsers.add_parser('export', help='Export session data')
        export_parser.add_argument('session_id', help='Session ID')
        export_parser.add_argument('--output', '-o', type=str, required=True, help='Output file')
        export_parser.add_argument('--format', choices=['json', 'pickle'], default='json', help='Export format')
        
        # Import session
        import_parser = session_subparsers.add_parser('import', help='Import session data')
        import_parser.add_argument('input_file', help='Input file to import')
        import_parser.add_argument('--session-id', help='New session ID (optional)')
        
        # Delete session
        delete_parser = session_subparsers.add_parser('delete', help='Delete session')
        delete_parser.add_argument('session_id', help='Session ID')
        
        # Save checkpoint
        checkpoint_parser = session_subparsers.add_parser('checkpoint', help='Save checkpoint')
        checkpoint_parser.add_argument('session_id', help='Session ID')
        checkpoint_parser.add_argument('checkpoint_name', help='Checkpoint name')
        
        # Load checkpoint
        load_parser = session_subparsers.add_parser('load', help='Load checkpoint')
        load_parser.add_argument('session_id', help='Session ID')
        load_parser.add_argument('--checkpoint', help='Checkpoint name (latest if not specified)')
    
    def _add_learning_commands(self, subparsers):
        """Add learning commands."""
        learn_parser = subparsers.add_parser('learn', help='Learn from text')
        learn_parser.add_argument('--text', type=str, help='Text to learn from')
        learn_parser.add_argument('--file', type=str, help='File containing text to learn from')
        learn_parser.add_argument('--session-id', type=str, help='Session ID')
        learn_parser.add_argument('--no-save', action='store_true', help='Do not save progress')
        learn_parser.add_argument('--batch-size', type=int, default=10, help='Batch size for file processing')
        learn_parser.add_argument('--consciousness', action='store_true', help='Enable consciousness integration')
        
        # Batch learning
        batch_learn_parser = subparsers.add_parser('learn-batch', help='Learn from multiple files')
        batch_learn_parser.add_argument('files', nargs='+', help='Files to learn from')
        batch_learn_parser.add_argument('--session-id', type=str, required=True, help='Session ID')
        batch_learn_parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
        batch_learn_parser.add_argument('--consciousness', action='store_true', help='Enable consciousness integration')
    
    def _add_tokenization_commands(self, subparsers):
        """Add tokenization commands."""
        tokenize_parser = subparsers.add_parser('tokenize', help='Tokenize text')
        tokenize_parser.add_argument('--text', type=str, help='Text to tokenize')
        tokenize_parser.add_argument('--file', type=str, help='File containing text to tokenize')
        tokenize_parser.add_argument('--session-id', type=str, help='Session ID')
        tokenize_parser.add_argument('--no-metadata', action='store_true', help='Exclude metadata from output')
        tokenize_parser.add_argument('--no-cache', action='store_true', help='Disable caching')
        tokenize_parser.add_argument('--consciousness', action='store_true', help='Enable consciousness integration')
        
        # Batch tokenization
        batch_tokenize_parser = subparsers.add_parser('tokenize-batch', help='Tokenize multiple files')
        batch_tokenize_parser.add_argument('files', nargs='+', help='Files to tokenize')
        batch_tokenize_parser.add_argument('--session-id', type=str, required=True, help='Session ID')
        batch_tokenize_parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
        batch_tokenize_parser.add_argument('--consciousness', action='store_true', help='Enable consciousness integration')
    
    def _add_analysis_commands(self, subparsers):
        """Add analysis commands."""
        analyze_parser = subparsers.add_parser('analyze', help='Analyze tokenization')
        analyze_parser.add_argument('--text', type=str, help='Text to analyze')
        analyze_parser.add_argument('--file', type=str, help='File containing text to analyze')
        analyze_parser.add_argument('--session-id', type=str, required=True, help='Session ID')
        analyze_parser.add_argument('--consciousness', action='store_true', help='Enable consciousness integration')
        
        # Cluster analysis
        clusters_parser = subparsers.add_parser('clusters', help='Analyze language clusters')
        clusters_parser.add_argument('--session-id', type=str, help='Session ID (all sessions if not specified)')
        clusters_parser.add_argument('--cluster-id', type=str, help='Specific cluster ID')
        
        # Statistics
        stats_parser = subparsers.add_parser('stats', help='Show statistics')
        stats_parser.add_argument('--session-id', type=str, help='Session ID (global if not specified)')
        stats_parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    
    def _add_comparison_commands(self, subparsers):
        """Add comparison commands."""
        if self.sentencepiece_adapter:
            compare_parser = subparsers.add_parser('compare', help='Compare with SentencePiece')
            compare_parser.add_argument('--text', type=str, help='Text to compare')
            compare_parser.add_argument('--file', type=str, help='File containing text to compare')
            compare_parser.add_argument('--session-id', type=str, required=True, help='Session ID')
            compare_parser.add_argument('--model-name', type=str, default='temp_model', help='SentencePiece model name')
            compare_parser.add_argument('--vocab-size', type=int, default=1000, help='Vocabulary size')
    
    def _add_utility_commands(self, subparsers):
        """Add utility commands."""
        # Health check
        subparsers.add_parser('health', help='System health check')
        
        # Clear caches
        subparsers.add_parser('clear-cache', help='Clear all caches')
        
        # Interactive mode
        interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
        interactive_parser.add_argument('--session-id', type=str, help='Session ID to use')
    
    async def run(self, args: List[str]) -> int:
        """
        Run CLI with provided arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            Exit code (0 for success, non-zero for error)
        """
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Set global options
        self.verbose = parsed_args.verbose
        self.output_format = parsed_args.output_format
        
        if parsed_args.session_id:
            self.current_session_id = parsed_args.session_id
        
        try:
            # Route to appropriate command handler
            if parsed_args.command == 'session':
                return await self._handle_session_command(parsed_args)
            elif parsed_args.command == 'learn':
                return await self._handle_learn_command(parsed_args)
            elif parsed_args.command == 'learn-batch':
                return await self._handle_learn_batch_command(parsed_args)
            elif parsed_args.command == 'tokenize':
                return await self._handle_tokenize_command(parsed_args)
            elif parsed_args.command == 'tokenize-batch':
                return await self._handle_tokenize_batch_command(parsed_args)
            elif parsed_args.command == 'analyze':
                return await self._handle_analyze_command(parsed_args)
            elif parsed_args.command == 'clusters':
                return await self._handle_clusters_command(parsed_args)
            elif parsed_args.command == 'stats':
                return await self._handle_stats_command(parsed_args)
            elif parsed_args.command == 'compare' and self.sentencepiece_adapter:
                return await self._handle_compare_command(parsed_args)
            elif parsed_args.command == 'health':
                return await self._handle_health_command(parsed_args)
            elif parsed_args.command == 'clear-cache':
                return await self._handle_clear_cache_command(parsed_args)
            elif parsed_args.command == 'interactive':
                return await self._handle_interactive_command(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            self._print_error("Operation cancelled by user")
            return 1
        except Exception as e:
            self._print_error(f"Error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    async def _handle_session_command(self, args) -> int:
        """Handle session management commands."""
        if args.session_command == 'create':
            tokenizer_config = {
                'max_clusters': args.max_clusters,
                'similarity_threshold': args.similarity_threshold,
                'learning_rate': args.learning_rate
            }
            
            session_id = await self.learning_service.create_learning_session(
                session_name=args.name,
                tokenizer_config=tokenizer_config
            )
            
            result = {
                'session_id': session_id,
                'session_name': args.name,
                'created_at': datetime.now().isoformat(),
                'config': tokenizer_config
            }
            
            self._print_output(result)
            return 0
        
        elif args.session_command == 'list':
            active_sessions = self.learning_service.get_active_sessions()
            result = {
                'active_sessions': active_sessions,
                'count': len(active_sessions)
            }
            self._print_output(result)
            return 0
        
        elif args.session_command == 'info':
            summary = await self.learning_service.get_session_summary(args.session_id)
            self._print_output(summary)
            return 0
        
        elif args.session_command == 'export':
            exported_data = await self.learning_service.export_session(args.session_id, args.format)
            
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                if args.format == 'json':
                    json.dump(exported_data, f, indent=2, ensure_ascii=False)
                else:
                    import pickle
                    pickle.dump(exported_data, f)
            
            self._print_info(f"Session exported to {output_path}")
            return 0
        
        elif args.session_command == 'import':
            input_path = Path(args.input_file)
            
            with open(input_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            imported_session_id = await self.learning_service.import_session(
                session_data, args.session_id
            )
            
            result = {
                'imported_session_id': imported_session_id,
                'source_file': str(input_path)
            }
            self._print_output(result)
            return 0
        
        elif args.session_command == 'delete':
            success = self.learning_service.close_session(args.session_id)
            result = {'session_id': args.session_id, 'deleted': success}
            self._print_output(result)
            return 0
        
        elif args.session_command == 'checkpoint':
            success = await self.learning_service.save_session_checkpoint(
                args.session_id, args.checkpoint_name
            )
            result = {
                'session_id': args.session_id,
                'checkpoint_name': args.checkpoint_name,
                'saved': success
            }
            self._print_output(result)
            return 0
        
        elif args.session_command == 'load':
            success = await self.learning_service.load_learning_session(
                args.session_id, args.checkpoint
            )
            result = {
                'session_id': args.session_id,
                'checkpoint': args.checkpoint,
                'loaded': success
            }
            self._print_output(result)
            return 0
        
        return 1
    
    async def _handle_learn_command(self, args) -> int:
        """Handle learning commands."""
        session_id = args.session_id or self.current_session_id
        if not session_id:
            self._print_error("Session ID required. Use --session-id or create a session first.")
            return 1
        
        # Get text from argument or file
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            self._print_error("Either --text or --file is required")
            return 1
        
        # Get consciousness state if requested
        consciousness_state = None
        if args.consciousness:
            consciousness_state = self._get_consciousness_state()
        
        # Learn from text
        result = await self.learning_service.learn_from_text(
            session_id=session_id,
            text=text,
            consciousness_state=consciousness_state,
            save_progress=not args.no_save
        )
        
        self._print_output(result)
        return 0
    
    async def _handle_learn_batch_command(self, args) -> int:
        """Handle batch learning commands."""
        texts = []
        for file_path in args.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        consciousness_state = None
        if args.consciousness:
            consciousness_state = self._get_consciousness_state()
        
        results = await self.learning_service.learn_from_batch(
            session_id=args.session_id,
            text_samples=texts,
            consciousness_state=consciousness_state,
            batch_size=args.batch_size
        )
        
        summary = {
            'total_files': len(args.files),
            'successful_learning': sum(1 for r in results if r.get('success', False)),
            'results': results
        }
        
        self._print_output(summary)
        return 0
    
    async def _handle_tokenize_command(self, args) -> int:
        """Handle tokenization commands."""
        session_id = args.session_id or self.current_session_id
        if not session_id:
            self._print_error("Session ID required")
            return 1
        
        # Get text
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            self._print_error("Either --text or --file is required")
            return 1
        
        consciousness_state = None
        if args.consciousness:
            consciousness_state = self._get_consciousness_state()
        
        result = await self.learning_service.tokenize_text(
            session_id=session_id,
            text=text,
            consciousness_state=consciousness_state,
            include_metadata=not args.no_metadata
        )
        
        self._print_output(result)
        return 0
    
    async def _handle_tokenize_batch_command(self, args) -> int:
        """Handle batch tokenization commands."""
        texts = []
        for file_path in args.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
        
        consciousness_state = None
        if args.consciousness:
            consciousness_state = self._get_consciousness_state()
        
        results = await self.learning_service.tokenize_batch(
            session_id=args.session_id,
            text_samples=texts,
            consciousness_state=consciousness_state,
            parallel_processing=args.parallel
        )
        
        summary = {
            'total_files': len(args.files),
            'successful_tokenizations': sum(1 for r in results if r.get('success', False)),
            'results': results
        }
        
        self._print_output(summary)
        return 0
    
    async def _handle_analyze_command(self, args) -> int:
        """Handle analysis commands."""
        session_id = args.session_id
        
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            self._print_error("Either --text or --file is required")
            return 1
        
        consciousness_state = None
        if args.consciousness:
            consciousness_state = self._get_consciousness_state()
        
        analysis = await self.learning_service.get_tokenization_analysis(
            session_id=session_id,
            text=text,
            consciousness_state=consciousness_state
        )
        
        self._print_output(analysis)
        return 0
    
    async def _handle_clusters_command(self, args) -> int:
        """Handle cluster analysis commands."""
        if args.session_id:
            clusters = await self.learning_service.get_language_clusters(args.session_id)
            result = {
                'session_id': args.session_id,
                'clusters': clusters
            }
        else:
            # Global cluster statistics
            stats = await self.repository.get_cluster_statistics()
            result = stats
        
        self._print_output(result)
        return 0
    
    async def _handle_stats_command(self, args) -> int:
        """Handle statistics commands."""
        if args.session_id:
            stats = await self.learning_service.get_session_summary(args.session_id)
        else:
            stats = await self.repository.get_cluster_statistics()
        
        self._print_output(stats)
        return 0
    
    async def _handle_compare_command(self, args) -> int:
        """Handle comparison commands."""
        if not self.sentencepiece_adapter:
            self._print_error("SentencePiece not available")
            return 1
        
        # Get text
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            self._print_error("Either --text or --file is required")
            return 1
        
        # Get custom tokenization
        tokenization_result = await self.learning_service.tokenize_text(
            session_id=args.session_id,
            text=text,
            include_metadata=True
        )
        
        custom_tokens = tokenization_result.get('tokens', [])
        
        # Train SentencePiece model if needed
        if args.model_name not in self.sentencepiece_adapter.list_models():
            self.sentencepiece_adapter.train_model_from_text(
                [text], args.model_name, args.vocab_size
            )
        
        # Compare
        comparison = self.sentencepiece_adapter.compare_tokenizations(
            text, custom_tokens, args.model_name
        )
        
        self._print_output(comparison)
        return 0
    
    async def _handle_health_command(self, args) -> int:
        """Handle health check command."""
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(self.learning_service.get_active_sessions()),
            'config': {
                'framework': self.config.framework.value,
                'device': self.config.device,
                'debug_mode': self.config.debug_mode
            }
        }
        
        self._print_output(health_info)
        return 0
    
    async def _handle_clear_cache_command(self, args) -> int:
        """Handle clear cache command."""
        cache_stats = self.learning_service.clear_cache()
        self._print_output(cache_stats)
        return 0
    
    async def _handle_interactive_command(self, args) -> int:
        """Handle interactive mode."""
        session_id = args.session_id or self.current_session_id
        
        self._print_info("Entering interactive mode. Type 'help' for commands, 'exit' to quit.")
        
        while True:
            try:
                user_input = input(f"multilingual({session_id or 'no-session'})> ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'help':
                    self._print_interactive_help()
                elif user_input.startswith('learn '):
                    text = user_input[6:]
                    if session_id:
                        result = await self.learning_service.learn_from_text(
                            session_id=session_id,
                            text=text,
                            save_progress=True
                        )
                        self._print_output(result)
                    else:
                        self._print_error("No active session. Create a session first.")
                elif user_input.startswith('tokenize '):
                    text = user_input[9:]
                    if session_id:
                        result = await self.learning_service.tokenize_text(
                            session_id=session_id,
                            text=text,
                            include_metadata=True
                        )
                        self._print_output(result)
                    else:
                        self._print_error("No active session. Create a session first.")
                elif user_input == 'sessions':
                    sessions = self.learning_service.get_active_sessions()
                    self._print_output({'active_sessions': sessions})
                elif user_input.startswith('use '):
                    new_session_id = user_input[4:]
                    session_id = new_session_id
                    self._print_info(f"Switched to session: {session_id}")
                else:
                    self._print_error("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._print_error(f"Error: {e}")
        
        self._print_info("Exiting interactive mode.")
        return 0
    
    def _print_interactive_help(self):
        """Print help for interactive mode."""
        help_text = """
Interactive Mode Commands:
  learn <text>           - Learn from text
  tokenize <text>        - Tokenize text
  sessions               - List active sessions
  use <session-id>       - Switch to session
  help                   - Show this help
  exit/quit              - Exit interactive mode
        """
        print(help_text)
    
    def _print_output(self, data: Any):
        """Print output in the specified format."""
        if self.output_format == 'json':
            print(json.dumps(data, indent=2, ensure_ascii=False))
        elif self.output_format == 'yaml':
            try:
                import yaml
                print(yaml.dump(data, default_flow_style=False, allow_unicode=True))
            except ImportError:
                print(json.dumps(data, indent=2, ensure_ascii=False))
        elif self.output_format == 'table':
            self._print_table(data)
        else:
            print(data)
    
    def _print_table(self, data: Any):
        """Print data in table format."""
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key:20} : {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                print(f"{i:3}: {item}")
        else:
            print(data)
    
    def _print_info(self, message: str):
        """Print info message."""
        if self.verbose:
            print(f"INFO: {message}", file=sys.stderr)
    
    def _print_error(self, message: str):
        """Print error message."""
        print(f"ERROR: {message}", file=sys.stderr)
    
    def _get_consciousness_state(self):
        """Get consciousness state for integration."""
        try:
            from ...domain.value_objects.consciousness_state import ConsciousnessState
            return ConsciousnessState.create_minimal_consciousness()
        except Exception:
            return None


def main():
    """Main entry point for CLI."""
    # Create default configuration
    config = SystemConfig.create_development_config() if '--debug' in sys.argv else SystemConfig()
    
    # Create CLI instance
    cli = MultilingualCLI(config)
    
    # Run CLI
    exit_code = asyncio.run(cli.run(sys.argv[1:]))
    sys.exit(exit_code)


if __name__ == '__main__':
    main()