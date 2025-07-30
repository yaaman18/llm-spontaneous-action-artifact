#!/usr/bin/env python3
"""
意識研究論文収集ツール
Google ScholarとarXivから意識研究に関する論文を検索し、PDFをダウンロードする
"""

import os
import time
import json
import requests
import feedparser
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import quote_plus, urljoin
from pathlib import Path
import logging
import re
from bs4 import BeautifulSoup
import random

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaperCollector:
    """論文収集メインクラス"""
    
    def __init__(self, output_dir: str = "collected_papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 各種ディレクトリの作成
        self.pdf_dir = self.output_dir / "pdfs"
        self.metadata_dir = self.output_dir / "metadata"
        self.pdf_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        # User-Agent設定（クローリング時の礼儀）
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 意識研究関連のキーワード
        self.consciousness_keywords = [
            "consciousness", "awareness", "phenomenology", "qualia",
            "integrated information theory", "IIT", "global workspace theory",
            "neural correlates consciousness", "NCC", "subjective experience",
            "self-awareness", "metacognition", "sentience", "hard problem",
            "binding problem", "phenomenal consciousness", "access consciousness"
        ]
    
    def search_arxiv(self, query: str, max_results: int = 50) -> List[Dict]:
        """arXiv APIを使用して論文を検索"""
        logger.info(f"Searching arXiv for: {query}")
        
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f"search_query=all:{quote_plus(query)}"
        params = f"{search_query}&start=0&max_results={max_results}"
        
        try:
            response = requests.get(base_url + params, headers=self.headers)
            response.raise_for_status()
            
            # フィードをパース
            feed = feedparser.parse(response.text)
            papers = []
            
            for entry in feed.entries:
                paper = {
                    'title': entry.title,
                    'authors': [author.name for author in entry.authors],
                    'abstract': entry.summary,
                    'arxiv_id': entry.id.split('/')[-1],
                    'pdf_url': entry.id.replace('abs', 'pdf') + '.pdf',
                    'published': entry.published,
                    'source': 'arXiv'
                }
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers on arXiv")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def search_google_scholar(self, query: str, num_results: int = 20) -> List[Dict]:
        """Google Scholarをスクレイピングして論文を検索（制限あり）"""
        logger.info(f"Searching Google Scholar for: {query}")
        
        # Google Scholarは直接のAPIを提供していないため、慎重にスクレイピング
        # 注意：頻繁なリクエストはブロックされる可能性があります
        
        papers = []
        base_url = "https://scholar.google.com/scholar"
        params = {
            'q': query,
            'hl': 'en',
            'num': num_results
        }
        
        try:
            # リクエスト間隔を設ける（礼儀正しいクローリング）
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 検索結果を解析
            for result in soup.find_all('div', class_='gs_r gs_or gs_scl'):
                try:
                    title_elem = result.find('h3', class_='gs_rt')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text()
                    
                    # PDFリンクを探す
                    pdf_link = None
                    pdf_elem = result.find('a', href=True, text=re.compile(r'\[PDF\]'))
                    if pdf_elem:
                        pdf_link = pdf_elem['href']
                    
                    # 著者情報
                    authors_elem = result.find('div', class_='gs_a')
                    authors_text = authors_elem.get_text() if authors_elem else ""
                    
                    # 要約
                    abstract_elem = result.find('div', class_='gs_rs')
                    abstract = abstract_elem.get_text() if abstract_elem else ""
                    
                    paper = {
                        'title': title,
                        'authors_text': authors_text,
                        'abstract': abstract,
                        'pdf_url': pdf_link,
                        'source': 'Google Scholar'
                    }
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Error parsing individual result: {e}")
                    continue
            
            logger.info(f"Found {len(papers)} papers on Google Scholar")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return []
    
    def download_pdf(self, pdf_url: str, filename: str) -> bool:
        """PDFをダウンロード"""
        if not pdf_url:
            return False
        
        try:
            logger.info(f"Downloading PDF from: {pdf_url}")
            
            response = requests.get(pdf_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # PDFかどうか確認
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(f"Not a PDF file: {content_type}")
                return False
            
            # ファイル保存
            filepath = self.pdf_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return False
    
    def save_metadata(self, paper: Dict, paper_id: str):
        """論文のメタデータを保存"""
        metadata_file = self.metadata_dir / f"{paper_id}.json"
        
        # タイムスタンプを追加
        paper['collected_at'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(paper, f, ensure_ascii=False, indent=2)
    
    def collect_consciousness_papers(self, max_papers_per_keyword: int = 10):
        """意識研究に関する論文を収集"""
        all_papers = []
        seen_titles = set()
        
        for keyword in self.consciousness_keywords:
            logger.info(f"\n--- Collecting papers for keyword: {keyword} ---")
            
            # arXivから検索
            arxiv_papers = self.search_arxiv(keyword, max_papers_per_keyword)
            
            for paper in arxiv_papers:
                # 重複チェック
                if paper['title'] in seen_titles:
                    continue
                
                seen_titles.add(paper['title'])
                all_papers.append(paper)
                
                # PDFダウンロード
                paper_id = f"arxiv_{paper['arxiv_id']}"
                pdf_filename = f"{paper_id}.pdf"
                
                if self.download_pdf(paper['pdf_url'], pdf_filename):
                    paper['pdf_file'] = pdf_filename
                
                # メタデータ保存
                self.save_metadata(paper, paper_id)
                
                # 礼儀正しいクローリング
                time.sleep(random.uniform(1, 2))
            
            # Google Scholarから検索（慎重に）
            # 注意：Google Scholarは頻繁なアクセスをブロックする可能性があります
            if len(all_papers) < max_papers_per_keyword * len(self.consciousness_keywords) / 2:
                scholar_papers = self.search_google_scholar(keyword, 5)
                
                for i, paper in enumerate(scholar_papers):
                    if paper['title'] in seen_titles:
                        continue
                    
                    seen_titles.add(paper['title'])
                    all_papers.append(paper)
                    
                    # PDFダウンロード（利用可能な場合）
                    paper_id = f"scholar_{keyword}_{i}"
                    pdf_filename = f"{paper_id}.pdf"
                    
                    if paper.get('pdf_url') and self.download_pdf(paper['pdf_url'], pdf_filename):
                        paper['pdf_file'] = pdf_filename
                    
                    # メタデータ保存
                    self.save_metadata(paper, paper_id)
                    
                    time.sleep(random.uniform(2, 4))
        
        # 収集結果のサマリーを作成
        summary = {
            'total_papers': len(all_papers),
            'collection_date': datetime.now().isoformat(),
            'keywords_used': self.consciousness_keywords,
            'papers_by_source': {
                'arXiv': len([p for p in all_papers if p.get('source') == 'arXiv']),
                'Google Scholar': len([p for p in all_papers if p.get('source') == 'Google Scholar'])
            }
        }
        
        with open(self.output_dir / 'collection_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n=== Collection Complete ===")
        logger.info(f"Total papers collected: {len(all_papers)}")
        logger.info(f"Results saved in: {self.output_dir}")
        
        return all_papers


def main():
    """メイン実行関数"""
    # 収集先ディレクトリを研究所のlibraryに設定
    output_dir = Path(__file__).parent.parent.parent / "library" / "collected_papers"
    
    collector = PaperCollector(str(output_dir))
    
    # 論文収集を実行
    papers = collector.collect_consciousness_papers(max_papers_per_keyword=5)
    
    print(f"\n収集完了！")
    print(f"収集した論文数: {len(papers)}")
    print(f"保存先: {output_dir}")


if __name__ == "__main__":
    main()