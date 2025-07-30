#!/usr/bin/env python3
"""
Maxwell Ramsteadの論文の完全パイプラインテスト
検索 → PDFダウンロード → マークダウン変換
"""

import requests
import json
import time
from pathlib import Path
from paper_collector import PaperCollector
from pdf_to_markdown import PDFToMarkdownConverter
from batch_convert import BatchPDFConverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_ramstead_pdfs():
    """Ramsteadの論文のPDFを直接ダウンロード"""
    print("🔍 Starting Ramstead paper collection and conversion pipeline...")
    
    # テスト用ディレクトリ設定
    test_dir = Path(__file__).parent / "ramstead_pipeline_test"
    pdf_dir = test_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Ramsteadの既知のarXiv論文リスト
    known_papers = [
        {
            'title': 'The inner screen model of consciousness',
            'arxiv_id': '2311.09206',
            'url': 'https://arxiv.org/pdf/2311.09206.pdf',
            'authors': ['MJD Ramstead', 'M Albarracin', 'A Kiefer', 'B Klein'],
            'year': '2023'
        },
        {
            'title': 'Is the free-energy principle a formal theory of semantics',
            'arxiv_id': '2006.10760', 
            'url': 'https://arxiv.org/pdf/2006.10760.pdf',
            'authors': ['MJD Ramstead', 'KJ Friston', 'I Hipólito'],
            'year': '2020'
        },
        {
            'title': 'Answering Schrödingers question: A free-energy formulation',
            'arxiv_id': '1705.07218',
            'url': 'https://arxiv.org/pdf/1705.07218.pdf', 
            'authors': ['MJD Ramstead', 'PB Badcock', 'KJ Friston'],
            'year': '2018'
        }
    ]
    
    downloaded_files = []
    
    print(f"📥 Downloading {len(known_papers)} Ramstead papers...")
    
    for i, paper in enumerate(known_papers, 1):
        try:
            print(f"\n--- Paper {i}: {paper['title'][:50]}... ---")
            
            # PDFファイル名
            pdf_filename = f"ramstead_{paper['arxiv_id']}.pdf"
            pdf_path = pdf_dir / pdf_filename
            
            print(f"📥 Downloading from: {paper['url']}")
            
            # PDFダウンロード
            response = requests.get(paper['url'], timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = pdf_path.stat().st_size / 1024 / 1024
            print(f"✅ Downloaded: {pdf_filename} ({file_size_mb:.1f} MB)")
            
            downloaded_files.append({
                'file_path': pdf_path,
                'paper_info': paper,
                'size_mb': file_size_mb
            })
            
            # メタデータも保存
            metadata = {
                'title': paper['title'],
                'authors': paper['authors'],
                'arxiv_id': paper['arxiv_id'],
                'year': paper['year'],
                'pdf_file': pdf_filename,
                'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_dir = test_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            with open(metadata_dir / f"{paper['arxiv_id']}.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            time.sleep(1)  # 礼儀正しいダウンロード
            
        except Exception as e:
            print(f"❌ Error downloading {paper['title']}: {e}")
            logger.error(f"Download error: {e}", exc_info=True)
    
    print(f"\n📊 Download Summary:")
    print(f"Papers downloaded: {len(downloaded_files)}")
    total_size = sum(f['size_mb'] for f in downloaded_files)
    print(f"Total size: {total_size:.1f} MB")
    
    return downloaded_files, test_dir


def convert_pdfs_to_markdown(downloaded_files, test_dir):
    """ダウンロードしたPDFをマークダウンに変換"""
    print(f"\n🔄 Converting {len(downloaded_files)} PDFs to Markdown...")
    
    pdf_dir = test_dir / "pdfs"
    markdown_dir = test_dir / "markdown"
    
    try:
        # PDF変換クラスを初期化
        converter = PDFToMarkdownConverter(str(pdf_dir), str(markdown_dir))
        
        # 全PDFを変換
        results = converter.convert_all_pdfs()
        
        print(f"\n📊 Conversion Results:")
        print(f"Total files: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['success']/results['total']*100:.1f}%" if results['total'] > 0 else "N/A")
        
        # 変換サマリーを取得
        summary = converter.get_conversion_summary()
        if summary:
            print(f"\n📈 Detailed Summary:")
            print(f"Success rate: {summary['success_rate']:.1f}%")
            print(f"Total pages converted: {summary['total_pages_converted']}")
            print(f"Average markdown length: {summary['average_markdown_length']:.0f} chars")
            
            if summary['failed_files']:
                print(f"Failed files: {summary['failed_files']}")
        
        return results, summary
        
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        logger.error(f"Conversion error: {e}", exc_info=True)
        return None, None


def check_conversion_quality(test_dir, downloaded_files):
    """変換結果の品質をチェック"""
    print(f"\n🔍 Checking conversion quality...")
    
    markdown_dir = test_dir / "markdown"
    
    if not markdown_dir.exists():
        print("❌ Markdown directory not found")
        return
    
    markdown_files = list(markdown_dir.glob("*.md"))
    print(f"Found {len(markdown_files)} markdown files")
    
    for md_file in markdown_files[:2]:  # 最初の2つをチェック
        print(f"\n--- Checking: {md_file.name} ---")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基本統計
            lines = content.split('\n')
            words = content.split()
            
            print(f"📄 File size: {len(content)} characters")
            print(f"📝 Lines: {len(lines)}")
            print(f"🔤 Words: {len(words)}")
            
            # メタデータの確認
            if content.startswith('---'):
                metadata_end = content.find('---', 3)
                if metadata_end > 0:
                    print("✅ Metadata header found")
                else:
                    print("⚠️ Incomplete metadata header")
            else:
                print("⚠️ No metadata header")
            
            # 数式の確認
            latex_count = content.count('$')
            if latex_count > 0:
                print(f"🧮 LaTeX expressions: ~{latex_count//2}")
            
            # 見出しの確認
            heading_count = len([line for line in lines if line.startswith('#')])
            print(f"📑 Headings: {heading_count}")
            
            # プレビュー表示
            preview_lines = lines[:15]
            print(f"\n📖 Preview (first 15 lines):")
            print("-" * 50)
            for line in preview_lines:
                print(line)
            print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error checking {md_file.name}: {e}")


def main():
    """メインの実行フロー"""
    try:
        print("🚀 Starting Maxwell Ramstead Paper Pipeline Test")
        print("=" * 60)
        
        # Step 1: PDFダウンロード
        downloaded_files, test_dir = download_ramstead_pdfs()
        
        if not downloaded_files:
            print("❌ No files downloaded, aborting pipeline")
            return
        
        # Step 2: PDF→マークダウン変換
        results, summary = convert_pdfs_to_markdown(downloaded_files, test_dir)
        
        if not results:
            print("❌ Conversion failed, aborting pipeline")
            return
        
        # Step 3: 品質チェック
        check_conversion_quality(test_dir, downloaded_files)
        
        # Step 4: 最終サマリー
        print(f"\n🎯 Pipeline Complete!")
        print("=" * 60)
        print(f"✅ Papers downloaded: {len(downloaded_files)}")
        print(f"✅ PDFs converted: {results['success']}/{results['total']}")
        print(f"📁 Output directory: {test_dir}")
        print(f"📊 Success rate: {results['success']/results['total']*100:.1f}%")
        
        # ファイル構造を表示
        print(f"\n📂 Generated files:")
        for item in sorted(test_dir.rglob("*")):
            if item.is_file():
                relative_path = item.relative_to(test_dir)
                size = item.stat().st_size
                if size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"   {relative_path} ({size_str})")
        
        return True
        
    except Exception as e:
        print(f"💥 Pipeline error: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")