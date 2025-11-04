from __future__ import annotations

from typing import Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json


router = APIRouter(prefix="/presets", tags=["presets"])


class PresetListItem(BaseModel):
    id: str
    name: str
    description: str | None = None
    tags: list[str] = []


def _examples_dir() -> Path:
    # backend/src/mg_api/routers/presets.py → repo root / modelTemplates
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    templates = repo_root / "modelTemplates"
    return templates


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/", response_model=List[PresetListItem])
def list_presets() -> List[PresetListItem]:
    d = _examples_dir()
    if not d.exists():
        return []
    items: list[PresetListItem] = []
    for p in sorted(d.glob("*.json")):
        try:
            data = _load_json(p)
            meta = data.get("metadata") or {}
            name = str(meta.get("name") or p.stem)
            desc = meta.get("description")
            tags = meta.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            items.append(PresetListItem(id=p.stem, name=name, description=desc, tags=[str(t) for t in tags]))
        except Exception:
            # If a file is malformed, still list it with filename
            items.append(PresetListItem(id=p.stem, name=p.stem))
    return items


@router.get("/{preset_id}")
def get_preset(preset_id: str) -> Any:
    d = _examples_dir()
    path = (d / f"{preset_id}.json").resolve()
    try:
        # ensure path within examples dir
        path.relative_to(d.resolve())
    except Exception:
        raise HTTPException(status_code=404, detail="not found")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    try:
        return _load_json(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to load preset: {e}")


