#!/bin/bash

# Consciousness Flow Visualizer - Setup Script
# 簡単セットアップスクリプト

echo "🌊 意識の流れビジュアライザー セットアップを開始します..."

# Python環境のチェック
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3が見つかりません。Python 3.8以上をインストールしてください。"
    exit 1
fi

# 仮想環境の作成
echo "📦 仮想環境を作成中..."
python3 -m venv venv

# 仮想環境の有効化
echo "🔧 仮想環境を有効化中..."
source venv/bin/activate

# 依存関係のインストール
echo "📚 依存関係をインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# 静的ファイルディレクトリの作成
echo "📁 必要なディレクトリを作成中..."
mkdir -p static

echo ""
echo "✅ セットアップ完了!"
echo ""
echo "使い方:"
echo "  1. 仮想環境を有効化: source venv/bin/activate"
echo "  2. サーバーを起動: python server.py"
echo "  3. ブラウザでアクセス: http://localhost:8080"
echo ""
echo "🎨 意識の流れを美しく可視化する準備が整いました！"