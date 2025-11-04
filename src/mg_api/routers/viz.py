from __future__ import annotations

from fastapi import APIRouter, Query

from ..store import store

# Import visualization builder/exporter from the kinetic model package
from kinetic_model.visualize import GraphSpecBuilder, CytoscapeExporter


router = APIRouter(prefix="/viz", tags=["viz"])


@router.get("/cytoscape")
def build_cytoscape_spec(
    microbiome_id: str | None = Query(default=None),
    metabolome_id: str | None = Query(default=None),
):
    """Return a Cytoscape.js spec for the current in-memory model.

    The model contains the saved metabolites only, under the key:
      { "metabolome": { "metabolites": { name: {...} } } }
    """
    # If a microbiome id is provided, build a full model (metabolome + microbiome)
    # using the stored runtime objects. If a metabolome id is provided, build the
    # model from that saved metabolome. Otherwise, show current metabolites only.
    builder = GraphSpecBuilder()
    if microbiome_id:
        try:
            model = store.get_microbiome_model(microbiome_id)
        except KeyError:
            # Fallback to empty
            model = {}
    elif metabolome_id:
        # Build model JSON from a saved metabolome in the store
        try:
            rec = store._get_metabolome_rec(metabolome_id)  # internal access ok here
            kmeta = rec["metabolome"]
            mets = {}
            for name in list(kmeta.metabolites):
                km = kmeta.get_metabolite(name)
                item = {
                    "id": name,
                    "name": name,
                    "concentration": float(km.concentration),
                    "formula": km.formula,
                    "color": km.color,
                }
                if getattr(km, "description", ""):
                    item["description"] = km.description
                mets[name] = item
            model = {"metabolome": {"metabolites": mets}}
        except KeyError:
            model = {}
    else:
        # Assemble minimal model dict from in-memory metabolites
        mets = {}
        for m in store.list_metabolites():
            item = {
                "id": m.name,
                "name": m.name,
                "concentration": m.concentration,
                "formula": m.formula,
                "color": m.color,
            }
            if m.description:
                item["description"] = m.description
            mets[m.name] = item
        model = {"metabolome": {"metabolites": mets}}

    spec = builder.build_from_json(model).build()
    cy = CytoscapeExporter().export(spec)
    return cy

