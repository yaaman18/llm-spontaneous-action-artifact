#!/usr/bin/env python3
"""
PDF to Markdown Converter
学術論文のPDFをマークダウン形式に変換する
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import traceback

# PDF処理ライブラリ
try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFToMarkdownConverter:
    """PDFからマークダウンへの変換クラス"""
    
    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = Path(input_dir)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.input_dir.parent / "markdown"
        
        self.output_dir.mkdir(exist_ok=True)
        
        # 変換ログ
        self.conversion_log = []
        self.log_file = self.output_dir / "conversion_log.json"
        
        # 使用可能なライブラリをチェック
        self.available_converters = self._check_available_converters()
        
        if not self.available_converters:
            raise RuntimeError("No PDF conversion libraries available. Please install pymupdf4llm or PyMuPDF.")
    
    def _check_available_converters(self) -> List[str]:
        """使用可能な変換ライブラリをチェック"""
        available = []
        
        if PYMUPDF4LLM_AVAILABLE:
            available.append("pymupdf4llm")
            logger.info("pymupdf4llm is available")
        
        if PYMUPDF_AVAILABLE:
            available.append("pymupdf")
            logger.info("PyMuPDF is available")
        
        return available
    
    def convert_with_pymupdf4llm(self, pdf_path: Path) -> Tuple[str, Dict]:
        """pymupdf4llmを使用してPDFを変換"""
        try:
            logger.info(f"Converting {pdf_path.name} with pymupdf4llm")
            
            # PDFをマークダウンに変換
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            
            # メタデータを抽出
            doc = fitz.open(str(pdf_path)) if PYMUPDF_AVAILABLE else None
            metadata = {}
            
            if doc:
                metadata = {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', ''),
                    'pages': doc.page_count,
                    'creation_date': doc.metadata.get('creationDate', ''),
                    'modification_date': doc.metadata.get('modDate', '')
                }
                doc.close()
            
            return md_text, metadata
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path.name} with pymupdf4llm: {e}")
            raise
    
    def convert_with_pymupdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """PyMuPDFを使用してPDFを変換（フォールバック）"""
        try:
            logger.info(f"Converting {pdf_path.name} with PyMuPDF")
            
            doc = fitz.open(str(pdf_path))
            
            # メタデータを抽出
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'pages': doc.page_count,
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', '')
            }
            
            # テキストを抽出
            md_text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # ページヘッダー
                md_text += f"\n## Page {page_num + 1}\n\n"
                
                # テキストブロックを抽出
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:  # テキストブロック
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # フォントサイズに基づいて見出しを判定
                                    font_size = span.get("size", 12)
                                    if font_size > 16:
                                        text = f"# {text}"
                                    elif font_size > 14:
                                        text = f"## {text}"
                                    elif font_size > 12:
                                        text = f"### {text}"
                                    
                                    line_text += text + " "
                            
                            if line_text.strip():
                                md_text += line_text.strip() + "\n\n"
                    
                    elif "image" in block:  # 画像ブロック
                        md_text += f"*[Image: {page_num + 1}]*\n\n"
            
            doc.close()
            return md_text, metadata
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path.name} with PyMuPDF: {e}")
            raise
    
    def convert_pdf(self, pdf_path: Path) -> bool:
        """単一のPDFファイルを変換"""
        try:
            logger.info(f"Starting conversion of {pdf_path.name}")
            
            # 出力ファイルパス
            output_file = self.output_dir / f"{pdf_path.stem}.md"
            
            # 変換処理
            md_text = ""
            metadata = {}
            conversion_method = ""
            
            # 優先順位で変換を試行
            if "pymupdf4llm" in self.available_converters:
                try:
                    md_text, metadata = self.convert_with_pymupdf4llm(pdf_path)
                    conversion_method = "pymupdf4llm"
                except Exception as e:
                    logger.warning(f"pymupdf4llm failed for {pdf_path.name}: {e}")
                    if "pymupdf" in self.available_converters:
                        md_text, metadata = self.convert_with_pymupdf(pdf_path)
                        conversion_method = "pymupdf"
                    else:
                        raise
            elif "pymupdf" in self.available_converters:
                md_text, metadata = self.convert_with_pymupdf(pdf_path)
                conversion_method = "pymupdf"
            
            # マークダウンファイルに保存
            with open(output_file, 'w', encoding='utf-8') as f:
                # メタデータヘッダーを追加
                f.write("---\n")
                f.write(f"title: {metadata.get('title', pdf_path.stem)}\n")
                f.write(f"author: {metadata.get('author', 'Unknown')}\n")
                f.write(f"pages: {metadata.get('pages', 'Unknown')}\n")
                f.write(f"conversion_method: {conversion_method}\n")
                f.write(f"converted_at: {datetime.now().isoformat()}\n")
                f.write("---\n\n")
                f.write(f"# {metadata.get('title', pdf_path.stem)}\n\n")
                f.write(md_text)
            
            # 変換ログに記録
            log_entry = {
                'pdf_file': str(pdf_path),
                'markdown_file': str(output_file),
                'conversion_method': conversion_method,
                'metadata': metadata,
                'converted_at': datetime.now().isoformat(),
                'success': True,
                'file_size_mb': round(pdf_path.stat().st_size / 1024 / 1024, 2),
                'markdown_length': len(md_text)
            }
            
            self.conversion_log.append(log_entry)
            logger.info(f"Successfully converted {pdf_path.name} to {output_file.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {pdf_path.name}: {e}")
            
            # エラーログに記録
            log_entry = {
                'pdf_file': str(pdf_path),
                'markdown_file': None,
                'conversion_method': None,
                'metadata': {},
                'converted_at': datetime.now().isoformat(),
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.conversion_log.append(log_entry)
            return False
    
    def convert_all_pdfs(self) -> Dict:
        """ディレクトリ内のすべてのPDFを変換"""
        logger.info(f"Converting all PDFs in {self.input_dir}")
        
        # PDFファイルを検索
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        success_count = 0
        
        for pdf_file in pdf_files:
            if self.convert_pdf(pdf_file):
                success_count += 1
        
        # 変換ログを保存
        self.save_conversion_log()
        
        results = {
            'total': len(pdf_files),
            'success': success_count,
            'failed': len(pdf_files) - success_count,
            'output_directory': str(self.output_dir)
        }
        
        logger.info(f"Conversion complete: {success_count}/{len(pdf_files)} successful")
        return results
    
    def save_conversion_log(self):
        """変換ログをファイルに保存"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversion_log, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Conversion log saved to {self.log_file}")
    
    def get_conversion_summary(self) -> Dict:
        """変換結果のサマリーを取得"""
        if not self.conversion_log:
            return {}
        
        successful = [log for log in self.conversion_log if log['success']]
        failed = [log for log in self.conversion_log if not log['success']]
        
        return {
            'total_files': len(self.conversion_log),
            'successful_conversions': len(successful),
            'failed_conversions': len(failed),
            'success_rate': len(successful) / len(self.conversion_log) * 100 if self.conversion_log else 0,
            'total_pages_converted': sum(log['metadata'].get('pages', 0) for log in successful),
            'average_markdown_length': sum(log['markdown_length'] for log in successful) / len(successful) if successful else 0,
            'failed_files': [log['pdf_file'] for log in failed]
        }


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PDFs to Markdown format')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('--output_dir', help='Output directory for markdown files')
    parser.add_argument('--single_file', help='Convert a single PDF file')
    
    args = parser.parse_args()
    
    try:
        converter = PDFToMarkdownConverter(args.input_dir, args.output_dir)
        
        if args.single_file:
            # 単一ファイル変換
            pdf_path = Path(args.single_file)
            if pdf_path.exists():
                success = converter.convert_pdf(pdf_path)
                if success:
                    print(f"✅ Successfully converted {pdf_path.name}")
                else:
                    print(f"❌ Failed to convert {pdf_path.name}")
            else:
                print(f"❌ File not found: {pdf_path}")
        else:
            # 全ファイル変換
            results = converter.convert_all_pdfs()
            
            print(f"\n📊 Conversion Results:")
            print(f"Total files: {results['total']}")
            print(f"Successful: {results['success']}")
            print(f"Failed: {results['failed']}")
            print(f"Output directory: {results['output_directory']}")
            
            # サマリー表示
            summary = converter.get_conversion_summary()
            if summary:
                print(f"\n📈 Summary:")
                print(f"Success rate: {summary['success_rate']:.1f}%")
                print(f"Total pages converted: {summary['total_pages_converted']}")
                if summary['failed_files']:
                    print(f"Failed files: {', '.join(Path(f).name for f in summary['failed_files'])}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()