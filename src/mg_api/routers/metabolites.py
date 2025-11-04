from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import MetaboliteIn, MetaboliteOut
from ..store import store


router = APIRouter(prefix="/metabolites", tags=["metabolites"])


@router.get("/", response_model=List[MetaboliteOut])
def list_metabolites() -> List[MetaboliteOut]:
    return store.list_metabolites()


@router.post("/", response_model=MetaboliteOut, status_code=status.HTTP_201_CREATED)
def create_metabolite(payload: MetaboliteIn) -> MetaboliteOut:
    return store.create_metabolite(payload)


@router.delete(
    "/{metabolite_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_metabolite(metabolite_id: str) -> Response:
    ok = store.delete_metabolite(metabolite_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metabolite not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
