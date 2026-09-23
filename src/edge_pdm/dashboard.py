from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"


class Telemetry(BaseModel):
    device_id: str = "unknown"
    timestamp: float
    prediction: str
    actual: Optional[str] = None
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float] = {}
    rms_g: float
    temperature_c: Optional[float] = None
    inference_ms: float
    features: list[float] = []


def create_app() -> FastAPI:
    app = FastAPI(title="Edge PDM Dashboard", version="1.0.0")
    app.state.history = deque(maxlen=300)
    app.state.clients = set()

    @app.get("/")
    async def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok", "samples": len(app.state.history)}

    @app.get("/api/history")
    async def history() -> list[dict]:
        return list(app.state.history)

    @app.post("/api/telemetry", status_code=202)
    async def telemetry(item: Telemetry) -> dict:
        payload = item.model_dump()
        app.state.history.append(payload)
        stale = []
        for client in app.state.clients:
            try:
                await client.send_text(json.dumps(payload))
            except Exception:
                stale.append(client)
        for client in stale:
            app.state.clients.discard(client)
        return {"accepted": True}

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        app.state.clients.add(websocket)
        try:
            while True:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            app.state.clients.discard(websocket)

    return app


app = create_app()
