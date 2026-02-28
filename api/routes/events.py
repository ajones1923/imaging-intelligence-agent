"""Event bus and DICOM webhook endpoints (Phase 2).

This module provides webhook endpoints for DICOM study events,
enabling real-time integration with PACS/Orthanc servers.
Currently a placeholder — will be activated in Phase 2 when
the Orthanc DICOM server is deployed.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from loguru import logger

events_router = APIRouter(prefix="/events", tags=["events"])


class StudyEvent(BaseModel):
    """DICOM study lifecycle event (Phase 2)."""
    event_type: str  # study.received, study.complete, study.updated
    study_uid: Optional[str] = None
    patient_id: Optional[str] = None
    modality: Optional[str] = None
    body_region: Optional[str] = None
    orthanc_id: Optional[str] = None


@events_router.post("/dicom-webhook")
async def dicom_webhook(event: StudyEvent):
    """Receive DICOM study events from Orthanc (Phase 2).

    This endpoint will be connected to Orthanc's webhook
    system to trigger automated imaging workflows when new
    studies arrive.
    """
    logger.info(f"DICOM webhook received: {event.event_type} (Phase 2 - not yet active)")
    return {
        "status": "accepted",
        "message": "DICOM event bus is Phase 2 — event logged but not processed",
        "event_type": event.event_type,
    }


@events_router.get("/status")
async def event_bus_status():
    """Check event bus status."""
    return {
        "phase": 2,
        "active": False,
        "message": "Event bus will be activated in Phase 2 with Orthanc DICOM server",
        "supported_events": [
            "study.received",
            "study.complete",
            "study.updated",
        ],
    }
