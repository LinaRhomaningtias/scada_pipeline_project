"""SCADA Pipeline Anomaly Detection - FastAPI entrypoint for Vercel and local use."""
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model.config import FEATURES, TARGET, THRESHOLD, TEAM, GITHUB_PROJECT_URL
from model.inference import predict, _METRICS

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
ASSETS_DIR = PUBLIC_DIR / "assets"
RECENT_FILE = ROOT / "outputs" / "recent.json"
RECENT = json.loads(RECENT_FILE.read_text(encoding="utf-8"))

app = FastAPI(title="SCADA Pipeline Anomaly Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Needed for local Uvicorn. On Vercel, public assets are also served by the CDN.
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

class PredictionInput(BaseModel):
    pressure: float = Field(...)
    flow_rate: float = Field(...)
    temperature: float = Field(...)
    valve_status: float = Field(...)
    pump_state: float = Field(...)
    pump_speed: float = Field(...)
    compressor_state: float = Field(...)
    energy_consumption: float = Field(...)

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(PUBLIC_DIR / "index.html")

@app.get("/api", include_in_schema=False)
def api_root():
    return {"message": "SCADA Pipeline Anomaly Detection API", "status": "online"}

@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "model": "MLP", "threshold": THRESHOLD}

@app.get("/metrics")
@app.get("/api/metrics")
def metrics():
    return {**_METRICS, "features": FEATURES, "target": TARGET}

@app.get("/recent")
@app.get("/api/recent")
def recent():
    return RECENT

@app.get("/team")
@app.get("/api/team")
def team():
    return {"github_project": GITHUB_PROJECT_URL, "members": TEAM}

@app.post("/predict")
@app.post("/api/predict")
def make_prediction(payload: PredictionInput):
    try:
        return predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
