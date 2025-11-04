from __future__ import annotations

from fastapi import APIRouter, HTTPException, status, Response

from ..schemas import SimulationRunIn, SimulationResultOut, SimulationListItem
from ..store import store


router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.post("/run", response_model=SimulationResultOut, status_code=status.HTTP_201_CREATED)
def run_simulation(payload: SimulationRunIn) -> SimulationResultOut:
    try:
        return store.run_simulation(payload)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/", response_model=list[SimulationListItem])
def list_simulations() -> list[SimulationListItem]:
    return store.list_simulations()  # type: ignore[return-value]


@router.get("/{simulation_id}", response_model=SimulationResultOut)
def get_simulation(simulation_id: str) -> SimulationResultOut:
    try:
        return store.get_simulation_plot(simulation_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation not found")


@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_simulation(simulation_id: str) -> Response:
    ok = store.delete_simulation(simulation_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Simulation not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)

