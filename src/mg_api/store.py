from __future__ import annotations

import threading
import uuid
from typing import Dict, List, Optional, Any

from .schemas import (
    MetaboliteIn,
    MetaboliteOut,
    MetabolomeIn,
    MetabolomeOut,
    PHFunctionCreate,
    PHFunctionOut,
    PHFunctionDetail,
    EnvironmentCreate,
    EnvironmentOut,
    EnvironmentDetail,
    PulseCreate,
    PulseOut,
    PulseDetail,
    # New: bacteria / microbiome
    FeedingTermIn,
    SubpopulationIn,
    TransitionIn,
    BacteriaIn,
    BacteriaOut,
    BacteriaRename,
    MicrobiomeCreate,
    MicrobiomeOut,
    # timelines / simulations
    TimelineCreate,
    TimelineOut,
    TimelineDetail,
    TimelineRename,
    SimulationRunIn,
    SimulationResultOut,
    SimulationListItem,
)

# Kinetic model domain objects
from kinetic_model.metabolite import Metabolite as KMetabolite
from kinetic_model.metabolome import Metabolome as KMetabolome
from kinetic_model.ph import pH as KPH
from kinetic_model.environment import Environment as KEnvironment
from kinetic_model.stirring import Stirring as KStirring
from kinetic_model.temperature import Temperature as KTemperature
import plotly.graph_objects as go


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metabolites: Dict[str, MetaboliteOut] = {}
        self._metabolomes: Dict[str, Dict[str, Any]] = {}
        self._ph_functions: Dict[str, Dict[str, Any]] = {}
        self._environments: Dict[str, Dict[str, Any]] = {}
        self._pulses: Dict[str, Dict[str, Any]] = {}
        self._timelines: Dict[str, Dict[str, Any]] = {}
        self._simulations: Dict[str, Dict[str, Any]] = {}
        # New: bacteria and microbiomes
        self._bacteria: Dict[str, Dict[str, Any]] = {}
        self._microbiomes: Dict[str, Dict[str, Any]] = {}

    # ---------------- Helpers ---------------- #
    def _unique_name(self, base: str, existing: set[str]) -> str:
        name = base
        i = 1
        while name in existing:
            name = f"{base}_{i}"
            i += 1
        return name

    # Metabolites CRUD
    def list_metabolites(self) -> List[MetaboliteOut]:
        with self._lock:
            return list(self._metabolites.values())

    def create_metabolite(self, payload: MetaboliteIn) -> MetaboliteOut:
        with self._lock:
            mid = uuid.uuid4().hex
            obj = MetaboliteOut(id=mid, **payload.model_dump())
            self._metabolites[mid] = obj
            return obj

    def delete_metabolite(self, metabolite_id: str) -> bool:
        with self._lock:
            return self._metabolites.pop(metabolite_id, None) is not None

    # Metabolomes CRUD (in-memory)
    def list_metabolomes(self) -> List[MetabolomeOut]:
        with self._lock:
            out: List[MetabolomeOut] = []
            for mid, rec in self._metabolomes.items():
                m: KMetabolome = rec["metabolome"]
                out.append(MetabolomeOut(id=mid, name=rec["name"], n_metabolites=len(m)))
            return out

    def create_metabolome(self, payload: MetabolomeIn) -> MetabolomeOut:
        with self._lock:
            # Build kinetic-model Metabolite objects from saved defs + requested concentrations
            saved_by_name: Dict[str, MetaboliteOut] = {m.name: m for m in self._metabolites.values()}
            kmets: List[KMetabolite] = []
            for name, conc in payload.concentrations.items():
                base = saved_by_name.get(name)
                if base is None:
                    raise ValueError(f"Unknown metabolite: {name}")
                kmets.append(
                    KMetabolite(
                        name=base.name,
                        concentration=float(conc),
                        formula=base.formula,
                        color=base.color,
                        description=base.description or "",
                    )
                )

            if not kmets:
                raise ValueError("At least one metabolite is required")

            kmeta = KMetabolome(kmets)
            mid = uuid.uuid4().hex
            self._metabolomes[mid] = {"name": payload.name, "metabolome": kmeta}
            return MetabolomeOut(id=mid, name=payload.name, n_metabolites=len(kmeta))

    def get_metabolome_plot(self, metabolome_id: str) -> Dict[str, Any]:
        with self._lock:
            rec = self._metabolomes.get(metabolome_id)
            if not rec:
                raise KeyError("metabolome not found")
            m: KMetabolome = rec["metabolome"]
            names = list(m.metabolites)
            concentrations = m.get_concentration()
            # Pull colors from internal metabolite objects
            colors = [m.get_metabolite(n).color for n in names]

            fig = go.Figure()
            for name, conc, color in zip(names, concentrations, colors):
                fig.add_trace(
                    go.Bar(
                        x=[name],
                        y=[float(conc)],
                        name=name,
                        marker=dict(color=color, line=dict(color="black", width=1.0)),
                        hovertemplate=f"<b>{name}</b><br>Concentration: {float(conc):.2f} mM<br><extra></extra>",
                        showlegend=True,
                    )
                )

            legend_config = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            if len(names) > 10:
                legend_config.update(dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))

            fig.update_layout(
                title=f"Metabolite Concentrations - {rec['name']}",
                xaxis_title="Metabolites",
                yaxis_title="Concentration (mM)",
                template="plotly_white",
                hovermode="closest",
                legend=legend_config,
                margin=dict(l=50, r=50, t=80, b=50),
            )
            fig.update_xaxes(showgrid=False, zeroline=False, tickangle=-45)
            fig.update_yaxes(zeroline=True, zerolinecolor="lightgray", gridcolor="lightgray", gridwidth=0.5)

            # Return a JSON-serializable dict (react-plotly expects data+layout)
            d = fig.to_dict()
            return {"data": d.get("data", []), "layout": d.get("layout", {}), "config": {"responsive": True}}

    def delete_metabolome(self, metabolome_id: str) -> bool:
        with self._lock:
            return self._metabolomes.pop(metabolome_id, None) is not None

    def rename_metabolome(self, metabolome_id: str, name: str) -> MetabolomeOut:
        with self._lock:
            rec = self._metabolomes.get(metabolome_id)
            if not rec:
                raise KeyError("metabolome not found")
            rec["name"] = name
            m: KMetabolome = rec["metabolome"]
            return MetabolomeOut(id=metabolome_id, name=name, n_metabolites=len(m))

    # ---------------- pH Functions ---------------- #
    def _get_metabolome_rec(self, metabolome_id: str) -> Dict[str, Any]:
        rec = self._metabolomes.get(metabolome_id)
        if not rec:
            raise KeyError("metabolome not found")
        return rec

    def list_ph_functions(self) -> List[PHFunctionOut]:
        with self._lock:
            out: List[PHFunctionOut] = []
            for fid, rec in self._ph_functions.items():
                m: KMetabolome = rec["metabolome"]
                out.append(
                    PHFunctionOut(
                        id=fid,
                        name=rec["name"],
                        metabolome_id=rec["metabolome_id"],
                        metabolome_name=rec["metabolome_name"],
                        n_metabolites=len(m),
                    )
                )
            return out

    def get_ph_function_detail(self, ph_id: str) -> PHFunctionDetail:
        with self._lock:
            rec = self._ph_functions.get(ph_id)
            if not rec:
                raise KeyError("ph function not found")
            return PHFunctionDetail(
                id=ph_id,
                name=rec["name"],
                metabolome_id=rec["metabolome_id"],
                baseValue=rec["baseValue"],
                weights=rec["weights"],
            )

    def create_ph_function(self, payload: PHFunctionCreate) -> PHFunctionOut:
        with self._lock:
            meta_rec = self._get_metabolome_rec(payload.metabolome_id)
            m: KMetabolome = meta_rec["metabolome"]
            # Validate weights keys exist in metabolome
            names_set = set(m.metabolites)
            invalid = set(payload.weights.keys()) - names_set
            if invalid:
                raise ValueError(f"Unknown metabolite(s) in weights: {sorted(invalid)}")
            # Clamp baseValue
            base = max(0.0, min(14.0, float(payload.baseValue)))

            # Build pH object bound to metabolome
            ph_obj = KPH(m, intercept=base, met_dictionary={k: float(v) for k, v in payload.weights.items()})

            fid = uuid.uuid4().hex
            self._ph_functions[fid] = {
                "name": payload.name,
                "metabolome_id": payload.metabolome_id,
                "metabolome_name": meta_rec["name"],
                "metabolome": m,
                "baseValue": base,
                "weights": {k: float(v) for k, v in payload.weights.items()},
                "ph_obj": ph_obj,
            }

            return PHFunctionOut(
                id=fid,
                name=payload.name,
                metabolome_id=payload.metabolome_id,
                metabolome_name=meta_rec["name"],
                n_metabolites=len(m),
            )

    def rename_ph_function(self, ph_id: str, name: str) -> PHFunctionOut:
        with self._lock:
            rec = self._ph_functions.get(ph_id)
            if not rec:
                raise KeyError("ph function not found")
            rec["name"] = name
            m: KMetabolome = rec["metabolome"]
            return PHFunctionOut(
                id=ph_id,
                name=name,
                metabolome_id=rec["metabolome_id"],
                metabolome_name=rec["metabolome_name"],
                n_metabolites=len(m),
            )

    def delete_ph_function(self, ph_id: str) -> bool:
        with self._lock:
            return self._ph_functions.pop(ph_id, None) is not None

    # ---------------- Environments ---------------- #
    def list_environments(self) -> List[EnvironmentOut]:
        with self._lock:
            out: List[EnvironmentOut] = []
            for eid, rec in self._environments.items():
                out.append(
                    EnvironmentOut(
                        id=eid,
                        name=rec["name"],
                        ph_function_id=rec["ph_function_id"],
                        ph_function_name=rec["ph_function_name"],
                        metabolome_id=rec["metabolome_id"],
                        metabolome_name=rec["metabolome_name"],
                        temperature=rec["temperature"],
                        stirring_rate=rec["stirring_rate"],
                        stirring_base_std=rec["stirring_base_std"],
                    )
                )
            return out

    def get_environment_detail(self, env_id: str) -> EnvironmentDetail:
        with self._lock:
            rec = self._environments.get(env_id)
            if not rec:
                raise KeyError("environment not found")
            return EnvironmentDetail(
                id=env_id,
                name=rec["name"],
                ph_function_id=rec["ph_function_id"],
                ph_function_name=rec["ph_function_name"],
                metabolome_id=rec["metabolome_id"],
                metabolome_name=rec["metabolome_name"],
                temperature=rec["temperature"],
                stirring_rate=rec["stirring_rate"],
                stirring_base_std=rec["stirring_base_std"],
            )

    def create_environment(self, payload: EnvironmentCreate) -> EnvironmentOut:
        with self._lock:
            ph_rec = self._ph_functions.get(payload.ph_function_id)
            if not ph_rec:
                raise KeyError("pH function not found")
            ph_obj: KPH = ph_rec["ph_obj"]
            metabolome: KMetabolome = ph_rec["metabolome"]

            # Normalize inputs via domain classes
            temp_obj = KTemperature(float(payload.temperature))
            stir_obj = KStirring(rate=float(payload.stirring_rate), base_std=float(payload.stirring_base_std))

            env_obj = KEnvironment(ph_obj, stir_obj, temp_obj)

            eid = uuid.uuid4().hex
            self._environments[eid] = {
                "name": payload.name,
                "ph_function_id": payload.ph_function_id,
                "ph_function_name": ph_rec["name"],
                "metabolome_id": ph_rec["metabolome_id"],
                "metabolome_name": ph_rec["metabolome_name"],
                "environment": env_obj,
                "temperature": temp_obj.temperature,
                "stirring_rate": stir_obj.rate,
                "stirring_base_std": stir_obj.base_std,
            }

            return EnvironmentOut(
                id=eid,
                name=payload.name,
                ph_function_id=payload.ph_function_id,
                ph_function_name=ph_rec["name"],
                metabolome_id=ph_rec["metabolome_id"],
                metabolome_name=ph_rec["metabolome_name"],
                temperature=temp_obj.temperature,
                stirring_rate=stir_obj.rate,
                stirring_base_std=stir_obj.base_std,
            )

    def rename_environment(self, env_id: str, name: str) -> EnvironmentOut:
        with self._lock:
            rec = self._environments.get(env_id)
            if not rec:
                raise KeyError("environment not found")
            rec["name"] = name
            return EnvironmentOut(
                id=env_id,
                name=name,
                ph_function_id=rec["ph_function_id"],
                ph_function_name=rec["ph_function_name"],
                metabolome_id=rec["metabolome_id"],
                metabolome_name=rec["metabolome_name"],
                temperature=rec["temperature"],
                stirring_rate=rec["stirring_rate"],
                stirring_base_std=rec["stirring_base_std"],
            )

    def delete_environment(self, env_id: str) -> bool:
        with self._lock:
            return self._environments.pop(env_id, None) is not None

    # ---------------- Pulses (metadata only; Reactor built later) ---------------- #
    def list_pulses(self) -> List[PulseOut]:
        with self._lock:
            out: List[PulseOut] = []
            for pid, rec in self._pulses.items():
                out.append(
                    PulseOut(
                        id=pid,
                        name=rec["name"],
                        t_start=rec["t_start"],
                        t_end=rec["t_end"],
                        n_steps=rec["n_steps"],
                        vin=rec["vin"],
                        vout=rec["vout"],
                        qin=rec["qin"],
                        qout=rec["qout"],
                        environment_id=rec["environment_id"],
                        environment_name=rec["environment_name"],
                        feed_metabolome_instant_id=rec.get("feed_metabolome_instant_id"),
                        feed_metabolome_instant_name=rec.get("feed_metabolome_instant_name"),
                        feed_metabolome_cont_id=rec.get("feed_metabolome_cont_id"),
                        feed_metabolome_cont_name=rec.get("feed_metabolome_cont_name"),
                        feed_microbiome_instant_id=rec.get("feed_microbiome_instant_id"),
                        feed_microbiome_instant_name=rec.get("feed_microbiome_instant_name"),
                        feed_microbiome_cont_id=rec.get("feed_microbiome_cont_id"),
                        feed_microbiome_cont_name=rec.get("feed_microbiome_cont_name"),
                    )
                )
            # Sort by start time to be nice
            out.sort(key=lambda x: x.t_start)
            return out

    def get_pulse_detail(self, pulse_id: str) -> PulseDetail:
        with self._lock:
            rec = self._pulses.get(pulse_id)
            if not rec:
                raise KeyError("pulse not found")
            return PulseDetail(
                id=pulse_id,
                name=rec["name"],
                t_start=rec["t_start"],
                t_end=rec["t_end"],
                n_steps=rec["n_steps"],
                vin=rec["vin"],
                vout=rec["vout"],
                qin=rec["qin"],
                qout=rec["qout"],
                environment_id=rec["environment_id"],
                environment_name=rec["environment_name"],
                feed_metabolome_instant_id=rec.get("feed_metabolome_instant_id"),
                feed_metabolome_instant_name=rec.get("feed_metabolome_instant_name"),
                feed_metabolome_cont_id=rec.get("feed_metabolome_cont_id"),
                feed_metabolome_cont_name=rec.get("feed_metabolome_cont_name"),
                feed_microbiome_instant_id=rec.get("feed_microbiome_instant_id"),
                feed_microbiome_instant_name=rec.get("feed_microbiome_instant_name"),
                feed_microbiome_cont_id=rec.get("feed_microbiome_cont_id"),
                feed_microbiome_cont_name=rec.get("feed_microbiome_cont_name"),
            )

    def create_pulse(self, payload: PulseCreate) -> PulseOut:
        with self._lock:
            # Validate environment
            env_rec = self._environments.get(payload.environment_id)
            if not env_rec:
                raise KeyError("environment not found")

            # Validate feeds (if provided)
            inst_name: Optional[str] = None
            cont_name: Optional[str] = None
            inst_mic_name: Optional[str] = None
            cont_mic_name: Optional[str] = None
            if payload.feed_metabolome_instant_id is not None:
                met_rec = self._metabolomes.get(payload.feed_metabolome_instant_id)
                if not met_rec:
                    raise KeyError("instant feed metabolome not found")
                inst_name = met_rec["name"]
            if payload.feed_metabolome_cont_id is not None:
                met_rec2 = self._metabolomes.get(payload.feed_metabolome_cont_id)
                if not met_rec2:
                    raise KeyError("continuous feed metabolome not found")
                cont_name = met_rec2["name"]
            # New: validate microbiome feeds
            if getattr(payload, "feed_microbiome_instant_id", None) is not None:
                mic_rec = self._microbiomes.get(payload.feed_microbiome_instant_id)  # type: ignore[attr-defined]
                if not mic_rec:
                    raise KeyError("instant feed microbiome not found")
                inst_mic_name = mic_rec["name"]
            if getattr(payload, "feed_microbiome_cont_id", None) is not None:
                mic_rec2 = self._microbiomes.get(payload.feed_microbiome_cont_id)  # type: ignore[attr-defined]
                if not mic_rec2:
                    raise KeyError("continuous feed microbiome not found")
                cont_mic_name = mic_rec2["name"]

            # Basic validation of times / steps / flows
            if payload.t_start >= payload.t_end:
                raise ValueError("t_start must be less than t_end")
            if payload.n_steps <= 0:
                raise ValueError("n_steps must be positive")
            for f in (payload.vin, payload.vout, payload.qin, payload.qout):
                if f < 0:
                    raise ValueError("flows must be non-negative")

            pid = uuid.uuid4().hex
            rec = {
                "name": payload.name,
                "t_start": float(payload.t_start),
                "t_end": float(payload.t_end),
                "n_steps": int(payload.n_steps),
                "vin": float(payload.vin),
                "vout": float(payload.vout),
                "qin": float(payload.qin),
                "qout": float(payload.qout),
                "environment_id": payload.environment_id,
                "environment_name": env_rec["name"],
                "feed_metabolome_instant_id": payload.feed_metabolome_instant_id,
                "feed_metabolome_instant_name": inst_name,
                "feed_metabolome_cont_id": payload.feed_metabolome_cont_id,
                "feed_metabolome_cont_name": cont_name,
                "feed_microbiome_instant_id": getattr(payload, "feed_microbiome_instant_id", None),
                "feed_microbiome_instant_name": inst_mic_name,
                "feed_microbiome_cont_id": getattr(payload, "feed_microbiome_cont_id", None),
                "feed_microbiome_cont_name": cont_mic_name,
            }
            self._pulses[pid] = rec

            return PulseOut(
                id=pid,
                name=rec["name"],
                t_start=rec["t_start"],
                t_end=rec["t_end"],
                n_steps=rec["n_steps"],
                vin=rec["vin"],
                vout=rec["vout"],
                qin=rec["qin"],
                qout=rec["qout"],
                environment_id=rec["environment_id"],
                environment_name=rec["environment_name"],
                feed_metabolome_instant_id=rec["feed_metabolome_instant_id"],
                feed_metabolome_instant_name=rec["feed_metabolome_instant_name"],
                feed_metabolome_cont_id=rec["feed_metabolome_cont_id"],
                feed_metabolome_cont_name=rec["feed_metabolome_cont_name"],
                feed_microbiome_instant_id=rec["feed_microbiome_instant_id"],
                feed_microbiome_instant_name=rec["feed_microbiome_instant_name"],
                feed_microbiome_cont_id=rec["feed_microbiome_cont_id"],
                feed_microbiome_cont_name=rec["feed_microbiome_cont_name"],
            )

    def rename_pulse(self, pulse_id: str, name: str) -> PulseOut:
        with self._lock:
            rec = self._pulses.get(pulse_id)
            if not rec:
                raise KeyError("pulse not found")
            rec["name"] = name
            return self.get_pulse_detail(pulse_id)

    def delete_pulse(self, pulse_id: str) -> bool:
        with self._lock:
            return self._pulses.pop(pulse_id, None) is not None

    # ---------------- Timelines ---------------- #
    def list_timelines(self) -> List[TimelineOut]:
        with self._lock:
            out: List[TimelineOut] = []
            for tid, rec in self._timelines.items():
                pulse_ids: List[str] = rec.get("pulse_ids", [])
                if pulse_ids:
                    ts = [self._pulses[pid]["t_start"] for pid in pulse_ids if pid in self._pulses]
                    te = [self._pulses[pid]["t_end"] for pid in pulse_ids if pid in self._pulses]
                    t0 = float(min(ts)) if ts else 0.0
                    t1 = float(max(te)) if te else 0.0
                else:
                    t0 = 0.0
                    t1 = 0.0
                out.append(TimelineOut(id=tid, name=rec["name"], n_pulses=len(pulse_ids), t_start=t0, t_end=t1))
            return out

    def create_timeline(self, payload: TimelineCreate) -> TimelineOut:
        with self._lock:
            # Validate pulse IDs exist
            for pid in payload.pulse_ids:
                if pid not in self._pulses:
                    raise KeyError(f"pulse not found: {pid}")
            tid = uuid.uuid4().hex
            self._timelines[tid] = {"name": payload.name, "pulse_ids": list(payload.pulse_ids)}
            # compute span
            ts = [self._pulses[pid]["t_start"] for pid in payload.pulse_ids] if payload.pulse_ids else []
            te = [self._pulses[pid]["t_end"] for pid in payload.pulse_ids] if payload.pulse_ids else []
            t0 = float(min(ts)) if ts else 0.0
            t1 = float(max(te)) if te else 0.0
            return TimelineOut(id=tid, name=payload.name, n_pulses=len(payload.pulse_ids), t_start=t0, t_end=t1)

    def get_timeline_detail(self, timeline_id: str) -> TimelineDetail:
        with self._lock:
            rec = self._timelines.get(timeline_id)
            if not rec:
                raise KeyError("timeline not found")
            out_pulses: List[PulseOut] = []
            for pid in rec.get("pulse_ids", []):
                if pid in self._pulses:
                    # reuse list_pulses mapping
                    pout_list = self.list_pulses()
                    match = next((p for p in pout_list if p.id == pid), None)
                    if match:
                        out_pulses.append(match)
            return TimelineDetail(id=timeline_id, name=rec["name"], pulses=out_pulses)

    def rename_timeline(self, timeline_id: str, name: str) -> TimelineOut:
        with self._lock:
            rec = self._timelines.get(timeline_id)
            if not rec:
                raise KeyError("timeline not found")
            rec["name"] = name
            pulse_ids: List[str] = rec.get("pulse_ids", [])
            ts = [self._pulses[pid]["t_start"] for pid in pulse_ids] if pulse_ids else []
            te = [self._pulses[pid]["t_end"] for pid in pulse_ids] if pulse_ids else []
            t0 = float(min(ts)) if ts else 0.0
            t1 = float(max(te)) if te else 0.0
            return TimelineOut(id=timeline_id, name=name, n_pulses=len(pulse_ids), t_start=t0, t_end=t1)

    def delete_timeline(self, timeline_id: str) -> bool:
        with self._lock:
            return self._timelines.pop(timeline_id, None) is not None

    # ---------------- Simulations ---------------- #
    def run_simulation(self, payload: SimulationRunIn) -> SimulationResultOut:
        from kinetic_model.reactor import Reactor, Pulse as KPulse
        # reuse already built KMetabolome and KMicrobiome from store state
        with self._lock:
            # metabolome via environment of timeline pulses: we rely on stored objects
            mic_rec = self._microbiomes.get(payload.microbiome_id)
            if not mic_rec:
                raise KeyError("microbiome not found")
            k_micro = mic_rec["microbiome"]

            # Find metabolome through env of first pulse (all envs share metabolome_id)
            tl = self._timelines.get(payload.timeline_id)
            if not tl:
                raise KeyError("timeline not found")
            pulse_ids: List[str] = tl.get("pulse_ids", [])
            if not pulse_ids:
                raise ValueError("timeline has no pulses")
            first_env_id = self._pulses[pulse_ids[0]]["environment_id"]
            env_rec = self._environments.get(first_env_id)
            if not env_rec:
                raise KeyError("environment not found")
            meta_id = env_rec["metabolome_id"]
            meta_rec = self._metabolomes.get(meta_id)
            if not meta_rec:
                raise KeyError("metabolome not found")
            k_meta = meta_rec["metabolome"]

            # Build Kinetic pulses in time order
            k_pulses: List[KPulse] = []
            for pid in sorted(pulse_ids, key=lambda x: self._pulses[x]["t_start"]):
                p = self._pulses[pid]
                env_id = p["environment_id"]
                env_rec2 = self._environments.get(env_id)
                if not env_rec2:
                    raise KeyError("environment not found")
                k_env = env_rec2["environment"]
                # Build feed metabolomes if provided
                inst_met = None
                cont_met = None
                if p.get("feed_metabolome_instant_id"):
                    mrec = self._metabolomes.get(p["feed_metabolome_instant_id"]) 
                    if mrec:
                        inst_met = mrec["metabolome"]
                if p.get("feed_metabolome_cont_id"):
                    mrec2 = self._metabolomes.get(p["feed_metabolome_cont_id"]) 
                    if mrec2:
                        cont_met = mrec2["metabolome"]
                # Microbiome feeds (optional)
                inst_mic = None
                cont_mic = None
                if p.get("feed_microbiome_instant_id"):
                    micr = self._microbiomes.get(p["feed_microbiome_instant_id"]) 
                    if micr:
                        inst_mic = micr["microbiome"]
                if p.get("feed_microbiome_cont_id"):
                    micr2 = self._microbiomes.get(p["feed_microbiome_cont_id"]) 
                    if micr2:
                        cont_mic = micr2["microbiome"]
                k_pulses.append(
                    KPulse(
                        t_start=p["t_start"], t_end=p["t_end"], n_steps=p["n_steps"],
                        vin=p["vin"], vout=p["vout"], qin=p["qin"], qout=p["qout"],
                        instant_feed_metabolome=inst_met,
                        instant_feed_microbiome=inst_mic,
                        continuous_feed_metabolome=cont_met,
                        continuous_feed_microbiome=cont_mic,
                        environment=k_env,
                    )
                )

            reactor = Reactor(k_micro, k_meta, k_pulses, float(payload.volume))
            mode = (payload.mode or "accurate").lower().strip()
            if mode == "fast":
                reactor.set_fast_simulation_mode()
            elif mode == "balanced":
                reactor.set_balanced_simulation_mode()
            # accurate: default settings
            reactor.simulate()

            # Build Plotly spec similar to make_plot()
            fig = reactor.make_plot()
            plot_dict = fig.to_dict()
            sid = uuid.uuid4().hex
            self._simulations[sid] = {
                "name": payload.name,
                "plot": plot_dict,
                "summary": {
                    "n_points": float(len(reactor.time_simul) if reactor.time_simul is not None else 0),
                    "n_metabolites": float(k_meta.nmets),
                },
            }
            return SimulationResultOut(id=sid, name=payload.name, summary=self._simulations[sid]["summary"], plot=plot_dict)

    def list_simulations(self) -> List[SimulationListItem]:
        with self._lock:
            return [SimulationListItem(id=sid, name=rec.get("name", sid)) for sid, rec in self._simulations.items()]

    def get_simulation_plot(self, simulation_id: str) -> SimulationResultOut:
        with self._lock:
            rec = self._simulations.get(simulation_id)
            if not rec:
                raise KeyError("simulation not found")
            return SimulationResultOut(id=simulation_id, name=rec.get("name", simulation_id), summary=rec.get("summary", {}), plot=rec.get("plot", {}))

    def delete_simulation(self, simulation_id: str) -> bool:
        with self._lock:
            return self._simulations.pop(simulation_id, None) is not None

 
    # ---------------- Bacteria ---------------- #
    def list_bacteria(self) -> List[BacteriaOut]:
        with self._lock:
            out: List[BacteriaOut] = []
            for bid, rec in self._bacteria.items():
                subpops = rec.get("subpopulations", [])
                names = [sp.get("name") for sp in subpops if isinstance(sp, dict)]
                out.append(BacteriaOut(id=bid, species=rec["species"], n_subpops=len(subpops), subpop_names=[n for n in names if n]))
            return out

    def create_bacteria(self, payload: BacteriaIn) -> BacteriaOut:
        with self._lock:
            # Enforce unique species name by suffixing _1, _2, ...
            existing_species = {rec["species"] for rec in self._bacteria.values()}
            species = self._unique_name(payload.species, existing_species)

            # Reject duplicate subpopulation names (exact match)
            provided_names = [sp.name for sp in (payload.subpopulations or [])]
            if len(provided_names) != len(set(provided_names)):
                raise ValueError("duplicate subpopulation names within species; names must be unique")

            # Build subpopulations with names as provided
            subpops: List[Dict[str, Any]] = []
            name_set: set[str] = set()
            for sp in payload.subpopulations:
                nm = sp.name
                name_set.add(nm)
                subpops.append({
                    "name": nm,
                    "species": species,
                    "count": float(sp.count),
                    "mumax": float(sp.mumax),
                    "pHopt": float(sp.pHopt),
                    "pH_sensitivity_left": float(sp.pH_sensitivity_left),
                    "pH_sensitivity_right": float(sp.pH_sensitivity_right),
                    "Topt": float(sp.Topt),
                    "tempSensitivity_left": float(sp.tempSensitivity_left),
                    "tempSensitivity_right": float(sp.tempSensitivity_right),
                    "state": sp.state,
                    "color": sp.color,
                    # Store raw feeding terms (metDict stays as provided lists)
                    "feedingTerms": [
                        {"id": (ft.id or f"{nm}_feeding"), "metDict": ft.metDict}
                        for ft in (sp.feedingTerms or [])
                    ],
                })

            # Validate and store connections as provided; ensure references exist
            connections: Dict[str, List[List[Any]]] = {}
            for src, lst in (payload.connections or {}).items():
                if src not in name_set:
                    # If a connection references an unknown source, reject to avoid dangling transitions
                    raise ValueError(f"connection source '{src}' not found among subpopulations")
                out_list: List[List[Any]] = []
                for tr in lst or []:
                    if tr.target not in name_set:
                        raise ValueError(f"connection target '{tr.target}' not found among subpopulations")
                    out_list.append([tr.target, tr.condition or "", float(tr.rate)])
                connections[src] = out_list

            bid = uuid.uuid4().hex
            self._bacteria[bid] = {
                "species": species,
                "color": payload.color,
                "subpopulations": subpops,
                "connections": connections,
            }
            return BacteriaOut(id=bid, species=species, n_subpops=len(subpops), subpop_names=[sp["name"] for sp in subpops])

    def delete_bacteria(self, bacteria_id: str) -> bool:
        with self._lock:
            return self._bacteria.pop(bacteria_id, None) is not None

    def rename_bacteria(self, bacteria_id: str, new_species: str) -> BacteriaOut:
        with self._lock:
            rec = self._bacteria.get(bacteria_id)
            if not rec:
                raise KeyError("bacteria not found")
            # Ensure uniqueness among other species
            existing = {r["species"] for bid2, r in self._bacteria.items() if bid2 != bacteria_id}
            new_name = self._unique_name(new_species, existing)
            rec["species"] = new_name
            # Update species for subpops as well
            for sp in rec.get("subpopulations", []):
                sp["species"] = new_name
            return BacteriaOut(id=bacteria_id, species=new_name, n_subpops=len(rec.get("subpopulations", [])), subpop_names=[sp["name"] for sp in rec.get("subpopulations", [])])

    # ---------------- Microbiomes ---------------- #
    def list_microbiomes(self) -> List[MicrobiomeOut]:
        with self._lock:
            out: List[MicrobiomeOut] = []
            for mid, rec in self._microbiomes.items():
                mic = rec["microbiome"]
                # Count subpops
                n_species = len(mic.bacteria)
                n_sub = sum(len(b.subpopulations) for b in mic.bacteria.values())
                out.append(MicrobiomeOut(id=mid, name=rec["name"], n_species=n_species, n_subpops=n_sub))
            return out

    def create_microbiome(self, payload: MicrobiomeCreate) -> MicrobiomeOut:
        from kinetic_model.metabolome import Metabolome as KMetabolome
        from kinetic_model.feeding_term import FeedingTerm as KFeeding
        from kinetic_model.subpopulation import Subpopulation as KSubpop
        from kinetic_model.bacteria import Bacteria as KBacteria
        from kinetic_model.microbiome import Microbiome as KMicro
        from kinetic_model.bacteria import evaluate_transition_condition

        with self._lock:
            # Validate metabolome
            mrec = self._metabolomes.get(payload.metabolome_id)
            if not mrec:
                raise KeyError("metabolome not found")
            kmet: KMetabolome = mrec["metabolome"]

            # Build bacteria objects from all saved bacteria
            bacteria_objs: Dict[str, Any] = {}
            # For fast lookup of metabolites
            meta_set = set(kmet.metabolites)

            for bid, bdef in self._bacteria.items():
                species = bdef["species"]
                color = bdef.get("color", "#54f542")

                # Build subpopulations
                subpop_objs: Dict[str, Any] = {}
                for sp in bdef.get("subpopulations", []):
                    sp_name = sp["name"]
                    # Counts: override from payload if provided
                    cnt = float(((payload.subpop_counts or {}).get(species, {}) or {}).get(sp_name, sp.get("count", 0.0)))
                    feeding_terms = []
                    for ft in sp.get("feedingTerms", []):
                        metd: Dict[str, list[float]] = ft.get("metDict", {}) or {}
                        # Validate metabolite names exist in selected metabolome
                        invalid = [mn for mn in metd.keys() if mn not in meta_set]
                        if invalid:
                            raise ValueError(f"Unknown metabolite(s) in feeding term for {sp_name}: {invalid}")
                        # Convert to FeedingTerm (expects tuple values)
                        converted: Dict[str, tuple[float, float]] = {}
                        for mn, pair in metd.items():
                            y = float(pair[0])
                            K = float(pair[1])
                            converted[mn] = (y, K)
                        feeding_terms.append(KFeeding(id=str(ft.get("id") or f"{sp_name}_feeding"), metDict=converted, metabolome=kmet))

                    subpop_objs[sp_name] = KSubpop(
                        name=sp_name,
                        count=cnt,
                        species=species,
                        mumax=float(sp.get("mumax", 0.0)),
                        feedingTerms=feeding_terms,
                        pHopt=float(sp.get("pHopt", 7.0)),
                        pH_sensitivity_left=float(sp.get("pH_sensitivity_left", 2.0)),
                        pH_sensitivity_right=float(sp.get("pH_sensitivity_right", 2.0)),
                        Topt=float(sp.get("Topt", 37.0)),
                        tempSensitivity_left=float(sp.get("tempSensitivity_left", 5.0)),
                        tempSensitivity_right=float(sp.get("tempSensitivity_right", 2.0)),
                        state=str(sp.get("state", "active")),
                        color=str(sp.get("color", "#aaaaaa")),
                    )

                # Build connections (leave conditions as strings; Bacteria resolves when used)
                connections: Dict[str, List[List[Any]]] = {}
                for src, lst in (bdef.get("connections", {}) or {}).items():
                    out_list: List[List[Any]] = []
                    for entry in lst or []:
                        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                            tgt, cond, rate = entry[0], entry[1], entry[2]
                            out_list.append([str(tgt), str(cond or ""), float(rate)])
                    connections[str(src)] = out_list

                bacteria_objs[species] = KBacteria(species, subpop_objs, connections, color=color, metabolome=kmet)

            # Unique microbiome name
            existing_names = {rec["name"] for rec in self._microbiomes.values()}
            mic_name = self._unique_name(payload.name, existing_names)
            mic_obj = KMicro(mic_name, bacteria_objs)
            mid = uuid.uuid4().hex
            self._microbiomes[mid] = {
                "name": mic_name,
                "microbiome": mic_obj,
                "metabolome_id": payload.metabolome_id,
            }

            n_species = len(bacteria_objs)
            n_sub = sum(len(b.subpopulations) for b in bacteria_objs.values())
            return MicrobiomeOut(id=mid, name=mic_name, n_species=n_species, n_subpops=n_sub)

    def rename_microbiome(self, microbiome_id: str, name: str) -> MicrobiomeOut:
        with self._lock:
            rec = self._microbiomes.get(microbiome_id)
            if not rec:
                raise KeyError("microbiome not found")
            existing = {r["name"] for mid2, r in self._microbiomes.items() if mid2 != microbiome_id}
            new_name = self._unique_name(name, existing)
            rec["name"] = new_name
            mic = rec["microbiome"]
            # Update object's name string repr if relevant
            try:
                mic.name = new_name  # type: ignore[attr-defined]
            except Exception:
                pass
            n_species = len(mic.bacteria)
            n_sub = sum(len(b.subpopulations) for b in mic.bacteria.values())
            return MicrobiomeOut(id=microbiome_id, name=new_name, n_species=n_species, n_subpops=n_sub)

    def delete_microbiome(self, microbiome_id: str) -> bool:
        with self._lock:
            return self._microbiomes.pop(microbiome_id, None) is not None

    def get_microbiome_model(self, microbiome_id: str) -> Dict[str, Any]:
        """Return full-model JSON dict for visualization (metabolome + microbiome)."""
        import json as _json
        with self._lock:
            rec = self._microbiomes.get(microbiome_id)
            if not rec:
                raise KeyError("microbiome not found")
            mic = rec["microbiome"]
            # Use to_json(full_model=True) to include metabolome block
            txt = mic.to_json(full_model=True)
            try:
                return _json.loads(txt)
            except Exception:
                return {}

    def get_microbiome_plot(self, microbiome_id: str) -> Dict[str, Any]:
        """Return a Plotly figure spec for a microbiome using make_plot()."""
        from kinetic_model.microbiome import Microbiome as KMicro
        with self._lock:
            rec = self._microbiomes.get(microbiome_id)
            if not rec:
                raise KeyError("microbiome not found")
            mic: KMicro = rec["microbiome"]
            fig = mic.make_plot(title=f"Microbiome Composition - {rec['name']}")
            d = fig.to_dict()
            return {"data": d.get("data", []), "layout": d.get("layout", {}), "config": {"responsive": True}}


store = InMemoryStore()
