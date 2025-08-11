#!/usr/bin/env python3
"""
意識状態モニタリングGUI

エナクティブ意識フレームワークの状態をリアルタイムで可視化する
日本語インターフェースです。
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime
import threading
import time
from typing import Dict, List, Optional
import sys

# 日本語フォントの設定
try:
    # macOSの日本語フォント設定
    if sys.platform == "darwin":  # macOS
        plt.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    # Windowsの日本語フォント設定  
    elif sys.platform == "win32":  # Windows
        plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    # Linuxの日本語フォント設定
    else:  # Linux
        plt.rcParams['font.family'] = ['Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP', 'DejaVu Sans']
    
    # 日本語の負の値表示を修正
    plt.rcParams['axes.unicode_minus'] = False
    
except Exception as e:
    print(f"日本語フォント設定警告: {e}")
    # フォント設定に失敗してもプログラムは続行
    plt.rcParams['axes.unicode_minus'] = False

# ドメインモジュールのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from domain.value_objects.phi_value import PhiValue
from domain.value_objects.consciousness_state import ConsciousnessState
from domain.value_objects.prediction_state import PredictionState
from domain.value_objects.probability_distribution import ProbabilityDistribution
from infrastructure.jax_predictive_coding_core import JaxPredictiveCodingCore


class ConsciousnessMonitor:
    """
    意識状態モニタリングGUI
    
    リアルタイムで意識状態、予測誤差、Φ値を可視化します。
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("エナクティブ意識フレームワーク - 状態モニター")
        self.root.geometry("1200x800")
        
        # データ保存用
        self.consciousness_history: List[float] = []
        self.phi_history: List[float] = []
        self.error_history: List[float] = []
        self.free_energy_history: List[float] = []
        self.timestamps: List[datetime] = []
        
        # 予測符号化コア
        self.predictive_core: Optional[JaxPredictiveCodingCore] = None
        
        # 実行制御
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        
        self.setup_gui()
        self.setup_predictive_core()
    
    def setup_gui(self):
        """GUIの設定"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 制御パネル
        control_frame = ttk.LabelFrame(main_frame, text="制御パネル", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="モニタリング開始", command=self.start_monitoring)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="モニタリング停止", command=self.stop_monitoring, state="disabled")
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.reset_button = ttk.Button(control_frame, text="データリセット", command=self.reset_data)
        self.reset_button.grid(row=0, column=2, padx=5)
        
        # 現在の状態表示
        status_frame = ttk.LabelFrame(main_frame, text="現在の意識状態", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # 状態指標
        self.phi_var = tk.StringVar(value="Φ値: --")\n        self.consciousness_var = tk.StringVar(value="意識レベル: --")
        self.error_var = tk.StringVar(value="予測誤差: --")
        self.free_energy_var = tk.StringVar(value="自由エネルギー: --")
        
        ttk.Label(status_frame, textvariable=self.phi_var, font=("Arial", 14)).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.consciousness_var, font=("Arial", 14)).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.error_var, font=("Arial", 14)).grid(row=2, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.free_energy_var, font=("Arial", 14)).grid(row=3, column=0, sticky=tk.W)
        
        # グラフ表示エリア
        self.setup_plots(main_frame)
        
        # ウィンドウの閉じる処理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_plots(self, parent):
        """グラフの設定"""
        # Matplotlibフィギュアの作成
        self.fig = Figure(figsize=(12, 6), dpi=100)
        self.fig.suptitle("意識状態の時系列変化", fontsize=16)
        
        # サブプロット
        self.ax1 = self.fig.add_subplot(221)
        self.ax1.set_title("Φ値")
        self.ax1.set_ylabel("Φ")
        self.ax1.grid(True)
        
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_title("意識レベル")
        self.ax2.set_ylabel("意識度")
        self.ax2.grid(True)
        
        self.ax3 = self.fig.add_subplot(223)
        self.ax3.set_title("予測誤差")
        self.ax3.set_ylabel("誤差")
        self.ax3.set_xlabel("時間")
        self.ax3.grid(True)
        
        self.ax4 = self.fig.add_subplot(224)
        self.ax4.set_title("自由エネルギー")
        self.ax4.set_ylabel("F")
        self.ax4.set_xlabel("時間")
        self.ax4.grid(True)
        
        # Tkinterウィジェットに埋め込み
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # レスポンシブ設定
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)
    
    def setup_predictive_core(self):
        """予測符号化コアの初期化"""
        try:
            import jax.random as random
            key = random.PRNGKey(42)
            
            # 3層の階層構造で初期化
            layer_dims = [10, 8, 6]  # 入力→中間→出力
            self.predictive_core = JaxPredictiveCodingCore(
                layer_dimensions=layer_dims,
                learning_rate=0.01,
                precision_adaptation_rate=0.001,
                random_key=key
            )
            print("予測符号化コア初期化完了")
        except ImportError:
            messagebox.showerror("エラー", "JAXライブラリが見つかりません。requirements.txtを確認してください。")
        except Exception as e:
            messagebox.showerror("エラー", f"予測符号化コア初期化失敗: {e}")
    
    def start_monitoring(self):
        """モニタリング開始"""
        if self.predictive_core is None:
            messagebox.showerror("エラー", "予測符号化コアが初期化されていません")
            return
        
        self.is_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # バックグラウンドでデータ更新
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
        
        # GUI更新
        self.schedule_gui_update()
    
    def stop_monitoring(self):
        """モニタリング停止"""
        self.is_running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
    
    def reset_data(self):
        """データリセット"""
        self.consciousness_history.clear()
        self.phi_history.clear()
        self.error_history.clear()
        self.free_energy_history.clear()
        self.timestamps.clear()
        
        # グラフをクリア
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        self.setup_plots_axes()
        self.canvas.draw()
    
    def setup_plots_axes(self):
        """グラフの軸設定"""
        self.ax1.set_title("Φ値")
        self.ax1.set_ylabel("Φ")
        self.ax1.grid(True)
        
        self.ax2.set_title("意識レベル")
        self.ax2.set_ylabel("意識度")
        self.ax2.grid(True)
        
        self.ax3.set_title("予測誤差")
        self.ax3.set_ylabel("誤差")
        self.ax3.set_xlabel("時間")
        self.ax3.grid(True)
        
        self.ax4.set_title("自由エネルギー")
        self.ax4.set_ylabel("F")
        self.ax4.set_xlabel("時間")
        self.ax4.grid(True)
    
    def update_loop(self):
        """バックグラウンドでのデータ更新ループ"""
        while self.is_running:
            try:
                # ランダムな入力データを生成（デモ用）
                input_data = np.random.randn(10) * 0.5 + 0.5
                
                # 予測符号化処理
                prediction_state = self.predictive_core.process_input(input_data)
                
                # 意識状態の構築
                consciousness_state = self.create_consciousness_state(prediction_state)
                
                # データ保存
                self.timestamps.append(datetime.now())
                self.phi_history.append(consciousness_state.phi_value.value)
                self.consciousness_history.append(consciousness_state.consciousness_level)
                self.error_history.append(prediction_state.total_error)
                
                # 自由エネルギーの取得
                free_energy = prediction_state.metadata.get('free_energy', 0.0)
                self.free_energy_history.append(float(free_energy))
                
                # データサイズ制限（最新100点）
                max_points = 100
                if len(self.timestamps) > max_points:
                    self.timestamps = self.timestamps[-max_points:]
                    self.phi_history = self.phi_history[-max_points:]
                    self.consciousness_history = self.consciousness_history[-max_points:]
                    self.error_history = self.error_history[-max_points:]
                    self.free_energy_history = self.free_energy_history[-max_points:]
                
                time.sleep(0.1)  # 100ms間隔
                
            except Exception as e:
                print(f"更新ループエラー: {e}")
                break
    
    def create_consciousness_state(self, prediction_state: PredictionState) -> ConsciousnessState:
        """予測状態から意識状態を構築"""
        # Φ値の計算（予測誤差から推定）
        phi_value = PhiValue(
            value=max(0.1, 1.0 - prediction_state.mean_error),
            complexity=1.0,
            integration=0.8
        )
        
        # 確率分布（不確実性）
        probs = np.array([0.6, 0.3, 0.1])  # デモ用固定値
        uncertainty_dist = ProbabilityDistribution(
            probabilities=probs,
            distribution_type="categorical"
        )
        
        # 意識状態の構築
        return ConsciousnessState(
            phi_value=phi_value,
            prediction_state=prediction_state,
            uncertainty_distribution=uncertainty_dist,
            metacognitive_confidence=0.7
        )
    
    def schedule_gui_update(self):
        """GUI更新のスケジュール"""
        if self.is_running:
            self.update_gui()
            self.root.after(200, self.schedule_gui_update)  # 200ms間隔でGUI更新
    
    def update_gui(self):
        """GUI表示の更新"""
        if not self.timestamps:
            return
        
        # 現在の値を表示
        latest_phi = self.phi_history[-1] if self.phi_history else 0
        latest_consciousness = self.consciousness_history[-1] if self.consciousness_history else 0
        latest_error = self.error_history[-1] if self.error_history else 0
        latest_free_energy = self.free_energy_history[-1] if self.free_energy_history else 0
        
        self.phi_var.set(f"Φ値: {latest_phi:.3f}")
        self.consciousness_var.set(f"意識レベル: {latest_consciousness:.3f}")
        self.error_var.set(f"予測誤差: {latest_error:.3f}")
        self.free_energy_var.set(f"自由エネルギー: {latest_free_energy:.3f}")
        
        # グラフ更新
        self.update_plots()
    
    def update_plots(self):
        """グラフの更新"""
        if len(self.timestamps) < 2:
            return
        
        # 時間軸（相対時間）
        start_time = self.timestamps[0]
        time_points = [(t - start_time).total_seconds() for t in self.timestamps]
        
        # グラフクリア
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # データプロット
        self.ax1.plot(time_points, self.phi_history, 'b-', linewidth=2)
        self.ax1.set_title("Φ値")
        self.ax1.set_ylabel("Φ")
        self.ax1.grid(True)
        
        self.ax2.plot(time_points, self.consciousness_history, 'g-', linewidth=2)
        self.ax2.set_title("意識レベル")
        self.ax2.set_ylabel("意識度")
        self.ax2.grid(True)
        
        self.ax3.plot(time_points, self.error_history, 'r-', linewidth=2)
        self.ax3.set_title("予測誤差")
        self.ax3.set_ylabel("誤差")
        self.ax3.set_xlabel("時間 (秒)")
        self.ax3.grid(True)
        
        self.ax4.plot(time_points, self.free_energy_history, 'm-', linewidth=2)
        self.ax4.set_title("自由エネルギー")
        self.ax4.set_ylabel("F")
        self.ax4.set_xlabel("時間 (秒)")
        self.ax4.grid(True)
        
        # レイアウト調整
        self.fig.tight_layout()
        self.canvas.draw()
    
    def on_closing(self):
        """ウィンドウ閉じる時の処理"""
        self.stop_monitoring()
        if self.update_thread and self.update_thread.is_alive():
            time.sleep(0.2)  # スレッド終了待ち
        self.root.destroy()
    
    def run(self):
        """GUIアプリケーション実行"""
        self.root.mainloop()


def main():
    """メイン関数"""
    try:
        app = ConsciousnessMonitor()
        app.run()
    except Exception as e:
        print(f"アプリケーション実行エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()