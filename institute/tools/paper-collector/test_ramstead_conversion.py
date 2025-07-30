#!/usr/bin/env python3
"""
Ramsteadの論文で実際のPDF変換テスト
arXivから直接論文をダウンロードして変換する
"""

import requests
import json
from pathlib import Path
from paper_collector import PaperCollector
from pdf_to_markdown import PDFToMarkdownConverter
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def search_and_convert_ramstead_papers():
    """Ramsteadの論文を検索してPDF変換を実行"""
    print("🔍 Searching for Ramstead papers with PDF links...")
    
    # テスト用出力ディレクトリ
    test_dir = Path(__file__).parent / "test_output" / "ramstead_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    collector = PaperCollector(str(test_dir))
    
    # より具体的なクエリでarXivを検索
    search_queries = [
        'au:"M Ramstead" OR au:"Maxwell Ramstead"',
        'all:"free energy principle" AND all:"Ramstead"',
        'all:"consciousness" AND all:"Ramstead"',
        'all:"active inference" AND all:"Ramstead"'
    ]
    
    all_papers = []
    seen_ids = set()
    
    for query in search_queries:
        print(f"\n🔎 Searching with query: {query}")
        papers = collector.search_arxiv(query, max_results=10)
        
        for paper in papers:
            if paper['arxiv_id'] not in seen_ids:
                all_papers.append(paper)
                seen_ids.add(paper['arxiv_id'])
        
        time.sleep(1)  # API制限を避けるため
    
    print(f"\n📚 Found {len(all_papers)} unique papers")
    
    if not all_papers:
        print("❌ No papers found. Testing with a known consciousness paper instead...")
        
        # 代替案：既知の意識研究論文を使用
        test_paper = {
            'title': 'The Inner Screen Model of Consciousness',
            'arxiv_id': '2301.11816',
            'pdf_url': 'https://arxiv.org/pdf/2301.11816.pdf',
            'authors': ['Test Author'],
            'abstract': 'Test consciousness paper'
        }
        all_papers = [test_paper]
    
    # PDFをダウンロードして変換
    successful_conversions = 0
    
    for i, paper in enumerate(all_papers[:3]):  # 最初の3つだけテスト
        print(f"\n--- Paper {i+1}: {paper['title'][:50]}... ---")
        
        try:
            # PDFダウンロード
            pdf_filename = f"paper_{paper['arxiv_id']}.pdf"
            pdf_path = test_dir / "pdfs" / pdf_filename
            pdf_path.parent.mkdir(exist_ok=True)
            
            print(f"📥 Downloading PDF from: {paper['pdf_url']}")
            
            response = requests.get(paper['pdf_url'], timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {pdf_path}")
            print(f"📄 Size: {pdf_path.stat().st_size / 1024:.1f} KB")
            
            # PDFをマークダウンに変換
            print(f"🔄 Converting to Markdown...")
            
            markdown_dir = test_dir / "markdown"
            converter = PDFToMarkdownConverter(str(pdf_path.parent), str(markdown_dir))
            
            success = converter.convert_pdf(pdf_path)
            
            if success:
                successful_conversions += 1
                print(f"✅ Conversion successful!")
                
                # マークダウンファイルをチェック
                markdown_file = markdown_dir / f"{pdf_path.stem}.md"
                if markdown_file.exists():
                    size_kb = markdown_file.stat().st_size / 1024
                    print(f"📝 Markdown created: {size_kb:.1f} KB")
                    
                    # プレビューを表示
                    with open(markdown_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        preview_lines = lines[:20]  # 最初の20行
                        print(f"\n📖 Preview:")
                        print("-" * 50)
                        for line in preview_lines:
                            print(line.rstrip())
                        print("-" * 50)
            else:
                print(f"❌ Conversion failed!")
        
        except Exception as e:
            print(f"❌ Error processing paper: {e}")
            logger.error(f"Error: {e}", exc_info=True)
        
        print()  # 空行
    
    print(f"\n🎯 Final Results:")
    print(f"Papers processed: {min(len(all_papers), 3)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Test output directory: {test_dir}")
    
    return successful_conversions > 0


if __name__ == "__main__":
    try:
        success = search_and_convert_ramstead_papers()
        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test failed!")
    except Exception as e:
        print(f"\n💥 Test error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)