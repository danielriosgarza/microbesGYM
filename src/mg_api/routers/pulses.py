from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import PulseCreate, PulseOut, PulseDetail, PulseRename
from ..store import store


router = APIRouter(prefix="/pulses", tags=["pulses"])


@router.get("/", response_model=List[PulseOut])
def list_pulses() -> List[PulseOut]:
    return store.list_pulses()


@router.post("/", response_model=PulseOut, status_code=status.HTTP_201_CREATED)
def create_pulse(payload: PulseCreate) -> PulseOut:
    try:
        return store.create_pulse(payload)
    except KeyError as e:
        msg = str(e)
        if "environment" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Environment not found")
        if "instant feed metabolome" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instant feed metabolome not found")
        if "continuous feed metabolome" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Continuous feed metabolome not found")
        if "instant feed microbiome" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Instant feed microbiome not found")
        if "continuous feed microbiome" in msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Continuous feed microbiome not found")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Related resource not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{pulse_id}", response_model=PulseDetail)
def get_pulse(pulse_id: str) -> PulseDetail:
    try:
        return store.get_pulse_detail(pulse_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pulse not found")


@router.patch("/{pulse_id}", response_model=PulseOut)
def rename_pulse(pulse_id: str, payload: PulseRename) -> PulseOut:
    try:
        return store.rename_pulse(pulse_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pulse not found")


@router.delete("/{pulse_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_pulse(pulse_id: str) -> Response:
    ok = store.delete_pulse(pulse_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pulse not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

