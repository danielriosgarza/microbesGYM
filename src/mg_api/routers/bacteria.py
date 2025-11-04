from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import BacteriaIn, BacteriaOut, BacteriaRename
from ..store import store


router = APIRouter(prefix="/bacteria", tags=["bacteria"])


@router.get("/", response_model=List[BacteriaOut])
def list_bacteria() -> List[BacteriaOut]:
    return store.list_bacteria()


@router.post("/", response_model=BacteriaOut, status_code=status.HTTP_201_CREATED)
def create_bacteria(payload: BacteriaIn) -> BacteriaOut:
    try:
        return store.create_bacteria(payload)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("/{bacteria_id}", response_model=BacteriaOut)
def rename_bacteria(bacteria_id: str, payload: BacteriaRename) -> BacteriaOut:
    try:
        return store.rename_bacteria(bacteria_id, payload.species)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bacteria not found")


@router.delete("/{bacteria_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_bacteria(bacteria_id: str) -> Response:
    ok = store.delete_bacteria(bacteria_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Bacteria not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

