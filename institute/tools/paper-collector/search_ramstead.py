#!/usr/bin/env python3
"""
Maxwell Ramsteadの論文を検索するスクリプト
"""

from paper_collector import PaperCollector
import json
from pathlib import Path
import time

def search_author_papers(author_name: str):
    """特定の著者の論文を検索"""
    
    # 出力ディレクトリを設定
    output_dir = Path(__file__).parent.parent.parent / "library" / "author_search" / author_name.replace(" ", "_")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    collector = PaperCollector(str(output_dir))
    
    print(f"\n=== Searching papers by {author_name} ===\n")
    
    # arXivで著者名検索（複数の検索パターンを試す）
    arxiv_papers = []
    
    # パターン1: フルネーム（スペースなし）
    arxiv_query = f'au:{author_name.replace(" ", "_")}'
    papers1 = collector.search_arxiv(arxiv_query, max_results=20)
    arxiv_papers.extend(papers1)
    
    # パターン2: 姓のみ
    time.sleep(1)  # リクエスト間隔
    last_name = author_name.split()[-1]
    arxiv_query2 = f'au:{last_name}'
    papers2 = collector.search_arxiv(arxiv_query2, max_results=30)
    
    # パターン3: 別の表記
    time.sleep(1)
    arxiv_query3 = f'au:Ramstead_M'
    papers3 = collector.search_arxiv(arxiv_query3, max_results=20)
    
    # 重複を除去
    seen_ids = {p['arxiv_id'] for p in arxiv_papers}
    for paper in papers2 + papers3:
        if paper['arxiv_id'] not in seen_ids:
            arxiv_papers.append(paper)
            seen_ids.add(paper['arxiv_id'])
    
    print(f"\nFound {len(arxiv_papers)} papers on arXiv")
    
    # Google Scholarでも検索
    print("\nSearching Google Scholar...")
    time.sleep(2)  # 礼儀正しいクローリング
    scholar_papers = collector.search_google_scholar(f'"{author_name}" consciousness OR "free energy" OR phenomenology', num_results=10)
    print(f"Found {len(scholar_papers)} papers on Google Scholar")
    
    # 結果を保存
    all_papers = arxiv_papers + scholar_papers
    
    for i, paper in enumerate(all_papers):
        print(f"\n--- Paper {i+1} ---")
        print(f"Title: {paper['title']}")
        if 'authors' in paper and isinstance(paper['authors'], list):
            print(f"Authors: {', '.join(paper['authors'])}")
        elif 'authors_text' in paper:
            print(f"Authors: {paper['authors_text']}")
        if 'published' in paper:
            print(f"Published: {paper['published']}")
        print(f"Abstract: {paper['abstract'][:200]}...")
        
        # メタデータ保存
        if 'arxiv_id' in paper:
            paper_id = f"arxiv_{paper['arxiv_id']}"
        else:
            paper_id = f"scholar_{i}"
        collector.save_metadata(paper, paper_id)
        
        # PDFダウンロード
        if 'pdf_url' in paper and paper['pdf_url']:
            pdf_filename = f"{paper_id}.pdf"
            if collector.download_pdf(paper['pdf_url'], pdf_filename):
                paper['pdf_file'] = pdf_filename
                print(f"PDF downloaded: {pdf_filename}")
    
    # サマリー保存
    summary = {
        'author': author_name,
        'total_papers': len(all_papers),
        'search_date': str(Path.cwd()),
        'papers': all_papers
    }
    
    with open(output_dir / 'search_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n=== Search Complete ===")
    print(f"Total papers found: {len(all_papers)}")
    print(f"Results saved in: {output_dir}")
    
    return all_papers


if __name__ == "__main__":
    # Maxwell Ramsteadの論文を検索（シンプルな名前で）
    papers = search_author_papers("Ramstead")