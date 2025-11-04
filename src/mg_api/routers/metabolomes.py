from __future__ import annotations

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import MetabolomeIn, MetabolomeOut, MetabolomeUpdate
from ..store import store


router = APIRouter(prefix="/metabolomes", tags=["metabolomes"])


@router.get("/", response_model=List[MetabolomeOut])
def list_metabolomes() -> List[MetabolomeOut]:
    return store.list_metabolomes()


@router.post("/", response_model=MetabolomeOut, status_code=status.HTTP_201_CREATED)
def create_metabolome(payload: MetabolomeIn) -> MetabolomeOut:
    try:
        return store.create_metabolome(payload)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{metabolome_id}/plot")
def metabolome_plot(metabolome_id: str) -> Dict[str, Any]:
    try:
        return store.get_metabolome_plot(metabolome_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metabolome not found")


@router.delete("/{metabolome_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_metabolome(metabolome_id: str) -> Response:
    ok = store.delete_metabolome(metabolome_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metabolome not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/{metabolome_id}", response_model=MetabolomeOut)
def update_metabolome(metabolome_id: str, payload: MetabolomeUpdate) -> MetabolomeOut:
    try:
        return store.rename_metabolome(metabolome_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metabolome not found")
