"""
Runs ON THE PC. Serves the dashboard, receives inference results pushed
from the Pi, and broadcasts them to connected browsers.
"""
import asyncio
import json
import queue
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"

ui_queue = queue.Queue()
stats = {
    'total_batches': 0,
    'faults_detected': 0,
    'suppressions_count': 0,
    'total_latency_ms': 0,
}


@app.get("/")
async def get_dashboard():
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.post("/ingest")
async def ingest_from_pi(request: Request):
    """The Pi POSTs each inference result here."""
    payload = await request.json()

    stats['total_batches'] += 1
    stats['total_latency_ms'] += payload.get('latency_ms', 0)
    if payload.get('status') == 'fault':
        stats['faults_detected'] += 1

    ui_queue.put_nowait(payload)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Browsers connect here for live dashboard updates."""
    await ws.accept()
    try:
        while True:
            try:
                payload = ui_queue.get_nowait()
                avg_latency = (stats['total_latency_ms'] / stats['total_batches']
                               if stats['total_batches'] > 0 else 0)
                payload['stats'] = {
                    'total_batches': stats['total_batches'],
                    'faults_detected': stats['faults_detected'],
                    'suppressions_count': stats.get('suppressions_count', 0),
                    'avg_latency_ms': round(avg_latency, 1),
                }
                await ws.send_text(json.dumps(payload))
            except queue.Empty:
                await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    print("Dashboard available at http://192.168.137.1:8000 (or this PC's LAN IP)")
    print("Waiting for the Pi to start pushing results...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")