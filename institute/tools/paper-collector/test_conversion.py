#!/usr/bin/env python3
"""
PDF変換機能のテストスクリプト
実際のPDFをダウンロードして変換をテストする
"""

import requests
import tempfile
from pathlib import Path
from pdf_to_markdown import PDFToMarkdownConverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_test_pdf(url: str, filename: str) -> Path:
    """テスト用PDFをダウンロード"""
    response = requests.get(url)
    response.raise_for_status()
    
    temp_dir = Path(tempfile.mkdtemp())
    pdf_path = temp_dir / filename
    
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    
    return pdf_path


def test_pdf_conversion():
    """PDF変換のテスト"""
    print("🧪 Testing PDF to Markdown conversion...")
    
    # arXivからテスト用PDFをダウンロード（意識研究関連の論文）
    test_pdfs = [
        {
            'url': 'https://arxiv.org/pdf/2301.11816.pdf',  # consciousness関連の論文
            'filename': 'consciousness_test.pdf',
            'title': 'Test Consciousness Paper'
        }
    ]
    
    for pdf_info in test_pdfs:
        try:
            print(f"\n📥 Downloading test PDF: {pdf_info['title']}")
            pdf_path = download_test_pdf(pdf_info['url'], pdf_info['filename'])
            
            print(f"✅ Downloaded to: {pdf_path}")
            print(f"📄 File size: {pdf_path.stat().st_size / 1024:.1f} KB")
            
            # 変換テスト
            print(f"\n🔄 Converting PDF to Markdown...")
            
            output_dir = pdf_path.parent / "markdown"
            converter = PDFToMarkdownConverter(str(pdf_path.parent), str(output_dir))
            
            success = converter.convert_pdf(pdf_path)
            
            if success:
                print(f"✅ Conversion successful!")
                
                # 結果ファイルを確認
                markdown_file = output_dir / f"{pdf_path.stem}.md"
                if markdown_file.exists():
                    file_size = markdown_file.stat().st_size
                    print(f"📝 Markdown file created: {markdown_file}")
                    print(f"📄 Markdown size: {file_size / 1024:.1f} KB")
                    
                    # 最初の数行を表示
                    with open(markdown_file, 'r', encoding='utf-8') as f:
                        content = f.read(500)
                        print(f"\n📖 Preview (first 500 chars):")
                        print("-" * 50)
                        print(content)
                        print("-" * 50)
            else:
                print(f"❌ Conversion failed!")
            
            # サマリーを表示
            summary = converter.get_conversion_summary()
            if summary:
                print(f"\n📊 Conversion Summary:")
                print(f"Success rate: {summary['success_rate']:.1f}%")
                print(f"Total pages: {summary['total_pages_converted']}")
                print(f"Average markdown length: {summary['average_markdown_length']:.1f} chars")
        
        except Exception as e:
            print(f"❌ Error testing {pdf_info['title']}: {e}")
            logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    test_pdf_conversion()