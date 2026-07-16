"""Webhook handler for ClawHub skill submission events."""
import logging
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhook", tags=["webhook"])

class ClawHubEvent(BaseModel):
    event_type: str  # "skill_submitted", "skill_updated"
    skill_name: str
    skill_url: str
    author: str
    timestamp: str

class WebhookResponse(BaseModel):
    accepted: bool
    message: str

@router.post("/clawhub", response_model=WebhookResponse)
async def handle_clawhub_webhook(event: ClawHubEvent):
    if event.event_type not in ("skill_submitted", "skill_updated"):
        raise HTTPException(status_code=400, detail=f"Unknown event type: {event.event_type}")

    logger.info(f"Received {event.event_type} for {event.skill_name} by {event.author}")

    # In production: clone skill, scan, report back
    # For now: accept and log
    return WebhookResponse(
        accepted=True,
        message=f"Scan queued for {event.skill_name}"
    )
