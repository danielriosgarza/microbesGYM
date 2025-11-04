from __future__ import annotations

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import MicrobiomeCreate, MicrobiomeOut, MicrobiomeRename
from ..store import store


router = APIRouter(prefix="/microbiomes", tags=["microbiomes"])


@router.get("/", response_model=List[MicrobiomeOut])
def list_microbiomes() -> List[MicrobiomeOut]:
    return store.list_microbiomes()


@router.post("/", response_model=MicrobiomeOut, status_code=status.HTTP_201_CREATED)
def create_microbiome(payload: MicrobiomeCreate) -> MicrobiomeOut:
    try:
        return store.create_microbiome(payload)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metabolome not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("/{microbiome_id}", response_model=MicrobiomeOut)
def rename_microbiome(microbiome_id: str, payload: MicrobiomeRename) -> MicrobiomeOut:
    try:
        return store.rename_microbiome(microbiome_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Microbiome not found")


@router.delete("/{microbiome_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_microbiome(microbiome_id: str) -> Response:
    ok = store.delete_microbiome(microbiome_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Microbiome not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{microbiome_id}/plot")
def microbiome_plot(microbiome_id: str) -> Dict[str, Any]:
    try:
        return store.get_microbiome_plot(microbiome_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Microbiome not found")
