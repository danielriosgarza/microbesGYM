from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import (
    TimelineCreate,
    TimelineOut,
    TimelineDetail,
    TimelineRename,
)
from ..store import store


router = APIRouter(prefix="/timelines", tags=["timelines"])


@router.get("/", response_model=List[TimelineOut])
def list_timelines() -> List[TimelineOut]:
    return store.list_timelines()


@router.post("/", response_model=TimelineOut, status_code=status.HTTP_201_CREATED)
def create_timeline(payload: TimelineCreate) -> TimelineOut:
    try:
        return store.create_timeline(payload)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{timeline_id}", response_model=TimelineDetail)
def get_timeline(timeline_id: str) -> TimelineDetail:
    try:
        return store.get_timeline_detail(timeline_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Timeline not found")


@router.patch("/{timeline_id}", response_model=TimelineOut)
def rename_timeline(timeline_id: str, payload: TimelineRename) -> TimelineOut:
    try:
        return store.rename_timeline(timeline_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Timeline not found")


@router.delete("/{timeline_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_timeline(timeline_id: str) -> Response:
    ok = store.delete_timeline(timeline_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Timeline not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


