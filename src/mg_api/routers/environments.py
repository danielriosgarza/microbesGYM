from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import EnvironmentCreate, EnvironmentOut, EnvironmentDetail, EnvironmentRename
from ..store import store


router = APIRouter(prefix="/environments", tags=["environments"])


@router.get("/", response_model=List[EnvironmentOut])
def list_environments() -> List[EnvironmentOut]:
    return store.list_environments()


@router.post("/", response_model=EnvironmentOut, status_code=status.HTTP_201_CREATED)
def create_environment(payload: EnvironmentCreate) -> EnvironmentOut:
    try:
        return store.create_environment(payload)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="pH function not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{env_id}", response_model=EnvironmentDetail)
def get_environment(env_id: str) -> EnvironmentDetail:
    try:
        return store.get_environment_detail(env_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Environment not found")


@router.patch("/{env_id}", response_model=EnvironmentOut)
def rename_environment(env_id: str, payload: EnvironmentRename) -> EnvironmentOut:
    try:
        return store.rename_environment(env_id, payload.name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Environment not found")


@router.delete("/{env_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_environment(env_id: str) -> Response:
    ok = store.delete_environment(env_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Environment not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

