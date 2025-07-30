#!/usr/bin/env python3
"""
論文収集ツールと連携する一括PDF変換スクリプト
ダウンロードした論文のPDFを自動的にマークダウンに変換する
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import json
import logging
from datetime import datetime

# 同じディレクトリのpdf_to_markdownをインポート
from pdf_to_markdown import PDFToMarkdownConverter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPDFConverter:
    """論文収集ツール用の一括PDF変換クラス"""
    
    def __init__(self, library_root: str = None):
        if library_root:
            self.library_root = Path(library_root)
        else:
            # デフォルトでは研究所のlibraryディレクトリを使用
            self.library_root = Path(__file__).parent.parent.parent / "library"
        
        self.processed_dirs = []
        self.conversion_results = []
    
    def find_pdf_directories(self) -> List[Path]:
        """PDFファイルが含まれるディレクトリを検索"""
        pdf_dirs = []
        
        # collected_papersディレクトリを探す
        collected_papers_dirs = list(self.library_root.rglob("collected_papers"))
        
        for collected_dir in collected_papers_dirs:
            pdf_dir = collected_dir / "pdfs"
            if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
                pdf_dirs.append(pdf_dir)
                logger.info(f"Found PDF directory: {pdf_dir}")
        
        # author_searchディレクトリも探す
        author_search_dirs = list(self.library_root.rglob("author_search"))
        
        for author_dir in author_search_dirs:
            for subdir in author_dir.iterdir():
                if subdir.is_dir():
                    pdf_dir = subdir / "pdfs"
                    if pdf_dir.exists() and list(pdf_dir.glob("*.pdf")):
                        pdf_dirs.append(pdf_dir)
                        logger.info(f"Found PDF directory: {pdf_dir}")
        
        return pdf_dirs
    
    def convert_directory(self, pdf_dir: Path) -> Dict:
        """単一ディレクトリのPDFを変換"""
        logger.info(f"Converting PDFs in {pdf_dir}")
        
        # マークダウン出力ディレクトリを設定
        markdown_dir = pdf_dir.parent / "markdown"
        
        try:
            converter = PDFToMarkdownConverter(str(pdf_dir), str(markdown_dir))
            results = converter.convert_all_pdfs()
            
            # 結果を記録
            result_entry = {
                'directory': str(pdf_dir),
                'markdown_directory': str(markdown_dir),
                'results': results,
                'summary': converter.get_conversion_summary(),
                'converted_at': datetime.now().isoformat()
            }
            
            self.conversion_results.append(result_entry)
            self.processed_dirs.append(pdf_dir)
            
            return result_entry
            
        except Exception as e:
            logger.error(f"Error converting directory {pdf_dir}: {e}")
            
            error_entry = {
                'directory': str(pdf_dir),
                'error': str(e),
                'converted_at': datetime.now().isoformat(),
                'success': False
            }
            
            self.conversion_results.append(error_entry)
            return error_entry
    
    def convert_all_found_pdfs(self) -> Dict:
        """見つかったすべてのPDFディレクトリを変換"""
        pdf_dirs = self.find_pdf_directories()
        
        if not pdf_dirs:
            logger.warning("No PDF directories found")
            return {
                'total_directories': 0,
                'processed_directories': 0,
                'total_files': 0,
                'successful_conversions': 0,
                'failed_conversions': 0
            }
        
        logger.info(f"Found {len(pdf_dirs)} directories with PDFs")
        
        total_files = 0
        total_successful = 0
        total_failed = 0
        
        for pdf_dir in pdf_dirs:
            result = self.convert_directory(pdf_dir)
            
            if 'results' in result:
                total_files += result['results']['total']
                total_successful += result['results']['success']
                total_failed += result['results']['failed']
        
        # 全体のサマリーを作成
        overall_summary = {
            'total_directories': len(pdf_dirs),
            'processed_directories': len(self.processed_dirs),
            'total_files': total_files,
            'successful_conversions': total_successful,
            'failed_conversions': total_failed,
            'success_rate': (total_successful / total_files * 100) if total_files > 0 else 0,
            'conversion_results': self.conversion_results,
            'processed_at': datetime.now().isoformat()
        }
        
        # 結果を保存
        self.save_batch_results(overall_summary)
        
        return overall_summary
    
    def save_batch_results(self, summary: Dict):
        """一括変換結果を保存"""
        results_file = self.library_root / "batch_conversion_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch conversion results saved to {results_file}")
    
    def convert_specific_author(self, author_name: str) -> Dict:
        """特定の著者のPDFを変換"""
        author_dir = self.library_root / "author_search" / author_name.replace(" ", "_")
        pdf_dir = author_dir / "pdfs"
        
        if not pdf_dir.exists():
            logger.error(f"Author directory not found: {author_dir}")
            return {'error': f'Author directory not found: {author_dir}'}
        
        if not list(pdf_dir.glob("*.pdf")):
            logger.warning(f"No PDF files found for author: {author_name}")
            return {'error': f'No PDF files found for author: {author_name}'}
        
        return self.convert_directory(pdf_dir)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch convert PDFs to Markdown for paper collection')
    parser.add_argument('--library_root', help='Root directory of the library')
    parser.add_argument('--author', help='Convert PDFs for specific author')
    parser.add_argument('--directory', help='Convert PDFs in specific directory')
    
    args = parser.parse_args()
    
    try:
        converter = BatchPDFConverter(args.library_root)
        
        if args.author:
            # 特定の著者のPDFを変換
            logger.info(f"Converting PDFs for author: {args.author}")
            result = converter.convert_specific_author(args.author)
            
            if 'error' in result:
                print(f"❌ {result['error']}")
            else:
                print(f"✅ Converted {result['results']['success']}/{result['results']['total']} files for {args.author}")
                
        elif args.directory:
            # 特定のディレクトリのPDFを変換
            pdf_dir = Path(args.directory)
            if not pdf_dir.exists():
                print(f"❌ Directory not found: {pdf_dir}")
                return
            
            result = converter.convert_directory(pdf_dir)
            if 'results' in result:
                print(f"✅ Converted {result['results']['success']}/{result['results']['total']} files in {pdf_dir}")
            else:
                print(f"❌ Failed to convert files in {pdf_dir}")
                
        else:
            # すべてのPDFを変換
            logger.info("Starting batch conversion of all found PDFs")
            summary = converter.convert_all_found_pdfs()
            
            print(f"\n📊 Batch Conversion Results:")
            print(f"Directories processed: {summary['processed_directories']}/{summary['total_directories']}")
            print(f"Total files: {summary['total_files']}")
            print(f"Successful conversions: {summary['successful_conversions']}")
            print(f"Failed conversions: {summary['failed_conversions']}")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            
            if summary['total_files'] > 0:
                print(f"\n📁 Converted directories:")
                for result in summary['conversion_results']:
                    if 'results' in result:
                        dir_name = Path(result['directory']).parent.name
                        print(f"  - {dir_name}: {result['results']['success']}/{result['results']['total']} files")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()