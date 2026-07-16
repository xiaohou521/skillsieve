"""SkillSieve REST API for ClawHub integration."""
import asyncio
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from skillsieve.core.pipeline import Pipeline

app = FastAPI(title="SkillSieve API", version="0.1.0")

# In-memory store for scan results (production would use a database)
_results_store: dict[str, dict] = {}
_report_store: dict[str, str] = {}

class ScanRequest(BaseModel):
    skill_path: str  # local path to skill directory
    max_layer: int = 3

class ScanResponse(BaseModel):
    verdict: str
    confidence: float
    layer_stopped: int
    latency_ms: float
    report_id: str | None = None

@app.post("/api/v1/scan", response_model=ScanResponse)
async def scan_skill(request: ScanRequest):
    path = Path(request.skill_path)
    if not path.exists() or not (path / "SKILL.md").exists():
        raise HTTPException(status_code=404, detail="Skill package not found")

    import time
    start = time.monotonic()
    pipeline = Pipeline(max_layer=request.max_layer)
    result = await pipeline.scan(path)
    latency = (time.monotonic() - start) * 1000

    report_id = None
    if result.report:
        import hashlib
        report_id = hashlib.md5(result.skill_name.encode()).hexdigest()[:12]
        _report_store[report_id] = result.report
        _results_store[report_id] = result.model_dump()

    return ScanResponse(
        verdict=result.final_verdict.value,
        confidence=result.final_confidence,
        layer_stopped=result.layer_stopped,
        latency_ms=round(latency, 1),
        report_id=report_id,
    )

@app.get("/api/v1/report/{report_id}")
async def get_report(report_id: str):
    if report_id not in _report_store:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report_id": report_id, "report": _report_store[report_id]}

@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}

from fastapi.responses import Response
from clawhub_integration.badge_generator import generate_badge


@app.get("/api/v1/badge/{skill_name}")
async def get_badge(skill_name: str):
    # Check if we have a cached result for this skill
    # If not, return a default "not scanned" badge
    # For now, just return a default badge
    svg = generate_badge("safe", 0.0, 0)

    # Check results store for this skill
    for rid, data in _results_store.items():
        if data.get("skill_name") == skill_name:
            svg = generate_badge(
                data["final_verdict"],
                data["final_confidence"],
                data["layer_stopped"],
            )
            break

    return Response(content=svg.encode("utf-8"), media_type="image/svg+xml; charset=utf-8")


from clawhub_integration.webhook_server import router as webhook_router
app.include_router(webhook_router)
