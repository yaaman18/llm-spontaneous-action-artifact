#!/usr/bin/env python3
"""
実際のRamsteadの意識研究論文での動作テスト
正確なarXiv IDを使用
"""

import requests
import json
import time
from pathlib import Path
from pdf_to_markdown import PDFToMarkdownConverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_real_ramstead_papers():
    """実際のRamsteadの意識研究論文をダウンロード"""
    print("🧠 Downloading actual Ramstead consciousness papers...")
    
    # テスト用ディレクトリ設定
    test_dir = Path(__file__).parent / "ramstead_consciousness_test"
    pdf_dir = test_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # 実際のRamsteadの論文（手動確認済み）
    actual_papers = [
        {
            'title': 'On Bayesian mechanics: a physics of and by beliefs',
            'arxiv_id': '2205.11543',
            'url': 'https://arxiv.org/pdf/2205.11543.pdf',
            'authors': ['Lancelot Da Costa', 'Karl Friston', 'Conor Heins', 'Maxwell J. D. Ramstead'],
            'year': '2022',
            'description': 'Bayesian mechanics and free energy principle'
        },
        {
            'title': 'Parcels and boundaries: A physics of and by beliefs',
            'arxiv_id': '2212.11989', 
            'url': 'https://arxiv.org/pdf/2212.11989.pdf',
            'authors': ['Lancelot Da Costa', 'Maxwell J. D. Ramstead', 'Karl Friston'],
            'year': '2022',
            'description': 'Extension of Bayesian mechanics'
        },
        {
            'title': 'Active inference and the analysis of action',
            'arxiv_id': '2201.07202',
            'url': 'https://arxiv.org/pdf/2201.07202.pdf', 
            'authors': ['Karl Friston', 'Maxwell J. D. Ramstead', 'et al.'],
            'year': '2022',
            'description': 'Active inference framework'
        }
    ]
    
    downloaded_files = []
    
    for i, paper in enumerate(actual_papers, 1):
        try:
            print(f"\n--- Paper {i}: {paper['title'][:60]}... ---")
            print(f"📝 Description: {paper['description']}")
            
            # PDFファイル名
            pdf_filename = f"ramstead_consciousness_{paper['arxiv_id']}.pdf"
            pdf_path = pdf_dir / pdf_filename
            
            print(f"📥 Downloading from: {paper['url']}")
            
            # PDFダウンロード
            response = requests.get(paper['url'], timeout=60)
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
                'description': paper['description'],
                'pdf_file': pdf_filename,
                'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_dir = test_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            
            with open(metadata_dir / f"{paper['arxiv_id']}.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            time.sleep(2)  # 礼儀正しいダウンロード
            
        except Exception as e:
            print(f"❌ Error downloading {paper['title']}: {e}")
            logger.error(f"Download error: {e}", exc_info=True)
    
    return downloaded_files, test_dir


def convert_and_analyze():
    """PDFをマークダウンに変換して分析"""
    print(f"\n🔄 Starting conversion and analysis...")
    
    downloaded_files, test_dir = download_real_ramstead_papers()
    
    if not downloaded_files:
        print("❌ No files downloaded")
        return
    
    # PDF変換
    pdf_dir = test_dir / "pdfs"
    markdown_dir = test_dir / "markdown"
    
    converter = PDFToMarkdownConverter(str(pdf_dir), str(markdown_dir))
    results = converter.convert_all_pdfs()
    
    print(f"\n📊 Conversion Results:")
    print(f"✅ Successful: {results['success']}/{results['total']}")
    print(f"📊 Success rate: {results['success']/results['total']*100:.1f}%")
    
    # 各論文の内容をプレビュー
    markdown_files = list(markdown_dir.glob("*.md"))
    
    for md_file in markdown_files:
        print(f"\n" + "="*60)
        print(f"📄 Analyzing: {md_file.name}")
        print("="*60)
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 基本統計
            lines = content.split('\n')
            words = content.split()
            
            print(f"📈 Statistics:")
            print(f"   Characters: {len(content):,}")
            print(f"   Lines: {len(lines):,}")
            print(f"   Words: {len(words):,}")
            
            # キーワード分析
            consciousness_keywords = [
                'consciousness', 'aware', 'experience', 'phenomeno', 
                'qualia', 'subjective', 'free energy', 'active inference',
                'Bayesian', 'belief', 'perception', 'action'
            ]
            
            keyword_counts = {}
            content_lower = content.lower()
            for keyword in consciousness_keywords:
                count = content_lower.count(keyword.lower())
                if count > 0:
                    keyword_counts[keyword] = count
            
            if keyword_counts:
                print(f"🧠 Consciousness-related keywords:")
                for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"   {keyword}: {count}")
            
            # 数式の確認
            math_count = content.count('$') + content.count('\\begin')
            if math_count > 0:
                print(f"🧮 Mathematical content: {math_count} expressions")
            
            # セクション構造
            sections = [line.strip() for line in lines if line.startswith('#')]
            print(f"📑 Sections found: {len(sections)}")
            if sections:
                print("   Main sections:")
                for section in sections[:5]:  # 最初の5つ
                    print(f"   - {section}")
            
            # アブストラクトの抽出
            abstract_start = -1
            abstract_end = -1
            for i, line in enumerate(lines):
                if 'abstract' in line.lower() and len(line) < 50:
                    abstract_start = i
                elif abstract_start > 0 and (line.startswith('#') or line.startswith('##')) and i > abstract_start + 3:
                    abstract_end = i
                    break
            
            if abstract_start > 0:
                if abstract_end == -1:
                    abstract_end = min(abstract_start + 20, len(lines))
                
                abstract_lines = lines[abstract_start:abstract_end]
                abstract_text = ' '.join(abstract_lines).strip()
                
                print(f"\n📖 Abstract preview:")
                print("-" * 50)
                print(abstract_text[:400] + "..." if len(abstract_text) > 400 else abstract_text)
                print("-" * 50)
            
        except Exception as e:
            print(f"❌ Error analyzing {md_file.name}: {e}")
    
    # 最終サマリー
    print(f"\n🎯 Final Summary:")
    print(f"✅ Papers processed: {len(downloaded_files)}")
    print(f"✅ PDFs converted: {results['success']}")
    print(f"📁 Output directory: {test_dir}")
    
    # ファイル一覧
    print(f"\n📂 Generated files:")
    for item in sorted(test_dir.rglob("*")):
        if item.is_file():
            relative_path = item.relative_to(test_dir)
            size = item.stat().st_size / 1024
            print(f"   {relative_path} ({size:.1f}KB)")


if __name__ == "__main__":
    try:
        convert_and_analyze()
        print("\n🎉 Consciousness paper analysis complete!")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        logger.error(f"Error: {e}", exc_info=True)