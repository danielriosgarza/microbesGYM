from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mg_api.routers.metabolites as metabolites
import mg_api.routers.viz as viz
import mg_api.routers.metabolomes as metabolomes
import mg_api.routers.ph_functions as ph_functions
import mg_api.routers.environments as environments
import mg_api.routers.pulses as pulses
import mg_api.routers.timelines as timelines
import mg_api.routers.simulations as simulations
import mg_api.routers.bacteria as bacteria
import mg_api.routers.microbiomes as microbiomes
import mg_api.routers.presets as presets


def create_app() -> FastAPI:
    app = FastAPI(title="Microbes Gym API", version="0.1.0")

    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # API routes
    app.include_router(metabolites.router, prefix="/api")
    app.include_router(viz.router, prefix="/api")
    app.include_router(metabolomes.router, prefix="/api")
    app.include_router(ph_functions.router, prefix="/api")
    app.include_router(environments.router, prefix="/api")
    app.include_router(pulses.router, prefix="/api")
    app.include_router(bacteria.router, prefix="/api")
    app.include_router(microbiomes.router, prefix="/api")
    app.include_router(timelines.router, prefix="/api")
    app.include_router(presets.router, prefix="/api")
    app.include_router(simulations.router, prefix="/api")

    return app


app = create_app()
