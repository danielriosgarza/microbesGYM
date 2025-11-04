from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import PHFunctionCreate, PHFunctionOut, PHFunctionDetail, PHFunctionRename
from ..store import store


router = APIRouter(prefix="/ph_functions", tags=["ph_functions"])


@router.get("/", response_model=List[PHFunctionOut])
def list_ph_functions() -> List[PHFunctionOut]:
    return store.list_ph_functions()


@router.post("/", response_model=PHFunctionOut, status_code=status.HTTP_201_CREATED)
def create_ph_function(payload: PHFunctionCreate) -> PHFunctionOut:
    try:
        return store.create_ph_function(payload)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metabolome not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{ph_id}", response_model=PHFunctionDetail)
def get_ph_function(ph_id: str) -> PHFunctionDetail:
    try:
        return store.get_ph_function_detail(ph_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH function not found")


@router.patch("/{ph_id}", response_model=PHFunctionOut)
def rename_ph_function(ph_id: str, payload: PHFunctionRename) -> PHFunctionOut:
    try:
        return store.rename_ph_function(ph_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH function not found")


@router.delete("/{ph_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_ph_function(ph_id: str) -> Response:
    ok = store.delete_ph_function(ph_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH function not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

