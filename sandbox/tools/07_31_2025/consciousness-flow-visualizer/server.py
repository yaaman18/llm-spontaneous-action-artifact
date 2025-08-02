"""
Consciousness Flow Visualizer - Web Server
リアルタイム可視化のためのWebサーバー
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import uvicorn
from typing import List
from consciousness_flow import ConsciousnessStream, ConsciousnessFlowGenerator, PhenomenalAnalyzer


app = FastAPI(title="Consciousness Flow Visualizer")

# グローバルな意識ストリーム
consciousness_stream = ConsciousnessStream(max_history=5000)
flow_generator = ConsciousnessFlowGenerator(consciousness_stream)

# 接続中のWebSocketクライアント
active_websockets: List[WebSocket] = []


@app.get("/")
async def get_index():
    """メインページを返す"""
    return HTMLResponse(content=open("static/index.html", "r").read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket接続を処理"""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # 現在の履歴を送信
        history = consciousness_stream.export_for_visualization(last_n=100)
        await websocket.send_json({
            "type": "history",
            "data": history
        })
        
        # リアルタイム更新を待機
        while True:
            # クライアントからのメッセージを待つ
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "get_analysis":
                # 現在の分析結果を送信
                dynamics = consciousness_stream.get_flow_dynamics()
                if consciousness_stream.current_state:
                    qualia = PhenomenalAnalyzer.analyze_qualia_structure(
                        consciousness_stream.current_state
                    )
                    dynamics["qualia_analysis"] = qualia
                    
                await websocket.send_json({
                    "type": "analysis",
                    "data": dynamics
                })
                
            elif message["type"] == "get_transitions":
                # 現象的遷移を検出して送信
                transitions = PhenomenalAnalyzer.detect_phenomenal_transitions(
                    consciousness_stream
                )
                await websocket.send_json({
                    "type": "transitions",
                    "data": transitions
                })
                
    except WebSocketDisconnect:
        active_websockets.remove(websocket)


async def broadcast_state(state_dict: dict):
    """全てのWebSocketクライアントに状態を配信"""
    for websocket in active_websockets:
        try:
            await websocket.send_json({
                "type": "update",
                "data": state_dict
            })
        except:
            # 接続が切れたクライアントを削除
            active_websockets.remove(websocket)


# 意識ストリームのオブザーバーとして登録
def state_observer(state):
    """新しい状態を全クライアントに配信"""
    asyncio.create_task(broadcast_state(state.to_dict()))

consciousness_stream.observers.append(state_observer)


@app.on_event("startup")
async def startup_event():
    """サーバー起動時の処理"""
    # テスト用のフロー生成を開始
    asyncio.create_task(flow_generator.start_generation())
    print("🌊 Consciousness Flow Visualizer started!")
    print("🎨 Access http://localhost:8081 to view the visualization")


@app.on_event("shutdown")
async def shutdown_event():
    """サーバー終了時の処理"""
    flow_generator.stop_generation()


# 静的ファイルを提供
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    # 静的ファイルディレクトリを作成
    import os
    os.makedirs("static", exist_ok=True)
    
    # サーバーを起動 (ポート8080が使用中の場合は8081を使用)
    uvicorn.run(app, host="0.0.0.0", port=8081)