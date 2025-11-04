from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import (
    PHDraftCreate,
    PHDraftOut,
    PHDraftDetail,
    PHDraftRename,
)
from ..store import store


router = APIRouter(prefix="/ph_drafts", tags=["ph_drafts"])


@router.get("/", response_model=List[PHDraftOut])
def list_ph_drafts() -> List[PHDraftOut]:
    return store.list_ph_drafts()


@router.post("/", response_model=PHDraftOut, status_code=status.HTTP_201_CREATED)
def create_ph_draft(payload: PHDraftCreate) -> PHDraftOut:
    try:
        return store.create_ph_draft(payload)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{draft_id}", response_model=PHDraftDetail)
def get_ph_draft(draft_id: str) -> PHDraftDetail:
    try:
        return store.get_ph_draft_detail(draft_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH draft not found")


@router.patch("/{draft_id}", response_model=PHDraftOut)
def rename_ph_draft(draft_id: str, payload: PHDraftRename) -> PHDraftOut:
    try:
        return store.rename_ph_draft(draft_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH draft not found")


@router.delete("/{draft_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_ph_draft(draft_id: str) -> Response:
    ok = store.delete_ph_draft(draft_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH draft not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


