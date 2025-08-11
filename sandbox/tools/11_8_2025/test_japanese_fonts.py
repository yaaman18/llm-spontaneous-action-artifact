#!/usr/bin/env python3
"""
日本語フォントテスト

利用可能な日本語フォントを確認し、matplotlibで日本語表示できるかテストします。
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import platform

def check_available_japanese_fonts():
    """利用可能な日本語フォントを確認"""
    print("=== 利用可能な日本語フォント確認 ===")
    print(f"OS: {platform.system()} {platform.release()}")
    
    # システムにインストールされているフォントを取得
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    # 日本語フォントを検索
    japanese_fonts = []
    japanese_keywords = ['Hiragino', 'Yu Gothic', 'Meiryo', 'MS Gothic', 'Takao', 'IPA', 'Noto Sans CJK', 'VL Gothic']
    
    for font_path in font_list:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            
            # 日本語フォントキーワードをチェック
            for keyword in japanese_keywords:
                if keyword.lower() in font_name.lower():
                    japanese_fonts.append((font_name, font_path))
                    break
        except Exception:
            continue
    
    if japanese_fonts:
        print(f"発見された日本語フォント: {len(japanese_fonts)}個")
        for name, path in japanese_fonts[:10]:  # 最初の10個を表示
            print(f"  - {name}")
    else:
        print("日本語フォントが見つかりませんでした")
    
    return japanese_fonts

def test_matplotlib_japanese():
    """matplotlibでの日本語表示テスト"""
    print("\n=== matplotlib日本語表示テスト ===")
    
    # 日本語フォント設定を試行
    if sys.platform == "darwin":  # macOS
        font_candidates = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
    elif sys.platform == "win32":  # Windows
        font_candidates = ['Yu Gothic', 'Meiryo', 'MS Gothic']
    else:  # Linux
        font_candidates = ['Takao', 'IPAexGothic', 'Noto Sans CJK JP']
    
    successful_font = None
    for font_name in font_candidates:
        try:
            plt.rcParams['font.family'] = [font_name]
            successful_font = font_name
            print(f"✓ {font_name} の設定に成功")
            break
        except Exception as e:
            print(f"✗ {font_name} の設定に失敗: {e}")
            continue
    
    if not successful_font:
        print("すべての日本語フォント設定が失敗しました")
        return False
    
    # 簡単な日本語プロットテスト
    try:
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'b-o')
        plt.title("日本語テスト - エナクティブ意識フレームワーク")
        plt.xlabel("時間 (秒)")
        plt.ylabel("Φ値")
        plt.grid(True, alpha=0.3)
        
        # ファイルに保存（表示はしない）
        plt.savefig('japanese_font_test.png', dpi=150, bbox_inches='tight')
        plt.close()  # 表示しないでクローズ
        
        print("✓ 日本語プロットの作成に成功")
        print("  'japanese_font_test.png' に保存されました")
        return True
        
    except Exception as e:
        print(f"✗ 日本語プロットの作成に失敗: {e}")
        return False

def recommend_font_setup():
    """フォント設定の推奨事項を表示"""
    print("\n=== フォント設定推奨事項 ===")
    
    if sys.platform == "darwin":  # macOS
        print("macOSでは通常、Hiragino SansまたはYu Gothicが利用可能です")
        print("追加インストールが必要な場合:")
        print("  brew install --cask font-noto-sans-cjk-jp")
        
    elif sys.platform == "win32":  # Windows
        print("WindowsではYu GothicまたはMeiryoが利用可能です")
        print("追加インストールが必要な場合:")
        print("  Noto Sans CJK JPをGoogleフォントからダウンロード")
        
    else:  # Linux
        print("Linuxでは以下のコマンドで日本語フォントをインストール:")
        print("Ubuntu/Debian: sudo apt install fonts-takao fonts-ipafont fonts-noto-cjk")
        print("CentOS/RHEL:   sudo yum install google-noto-sans-cjk-jp-fonts")
        print("Arch:          sudo pacman -S noto-fonts-cjk")

def main():
    """メイン関数"""
    print("日本語フォント設定テスト開始\n")
    
    # 1. 利用可能な日本語フォントを確認
    japanese_fonts = check_available_japanese_fonts()
    
    # 2. matplotlib日本語表示テスト
    test_success = test_matplotlib_japanese()
    
    # 3. 推奨事項表示
    recommend_font_setup()
    
    # 4. 結果サマリー
    print(f"\n=== テスト結果サマリー ===")
    print(f"発見された日本語フォント数: {len(japanese_fonts)}")
    print(f"matplotlib日本語表示: {'✓ 成功' if test_success else '✗ 失敗'}")
    
    if test_success:
        print("\n✓ 日本語表示が正常に設定されました")
        print("エナクティブ意識フレームワークのデモを実行できます:")
        print("  python demo.py")
        print("  python main.py --gui")
    else:
        print("\n✗ 日本語表示の設定に問題があります")
        print("上記の推奨事項に従ってフォントをインストールしてください")
    
    return test_success

if __name__ == "__main__":
    main()