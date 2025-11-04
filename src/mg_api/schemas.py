from __future__ import annotations

from typing import Dict, Optional, List, Any

from pydantic import BaseModel, Field, field_validator


SUPPORTED_ELEMENTS = {"C", "H", "O", "N", "S", "P"}


class MetaboliteIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    concentration: float = Field(0.0, ge=0.0, description="Concentration in mM")
    formula: Dict[str, int] = Field(
        default_factory=dict,
        description="Element counts, keys in {C,H,O,N,S,P}",
    )
    color: str = Field("#0093f5", min_length=1)
    description: Optional[str] = Field(default="")

    @field_validator("formula")
    @classmethod
    def validate_formula(cls, v: Dict[str, int]) -> Dict[str, int]:
        if not isinstance(v, dict):
            raise ValueError("formula must be an object")
        invalid = set(v.keys()) - SUPPORTED_ELEMENTS
        if invalid:
            raise ValueError(f"Unsupported elements: {sorted(invalid)}")
        norm: Dict[str, int] = {}
        for k, val in v.items():
            if not isinstance(val, int):
                raise ValueError(f"Element '{k}' must be an integer")
            if val < 0:
                raise ValueError(f"Element '{k}' cannot be negative")
            norm[k] = val
        # ensure all supported present with default 0 for convenience
        for k in SUPPORTED_ELEMENTS:
            norm.setdefault(k, 0)
        return norm


class MetaboliteOut(MetaboliteIn):
    id: str


# ---------------- Metabolomes ---------------- #
class MetabolomeIn(BaseModel):
    name: str
    concentrations: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of metabolite name -> initial concentration (mM)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v

    @field_validator("concentrations")
    @classmethod
    def validate_concs(cls, m: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, val in (m or {}).items():
            try:
                f = float(val)
            except Exception as e:
                raise ValueError(f"invalid concentration for {k!r}") from e
            if f < 0:
                f = 0.0
            out[str(k)] = f
        return out


class MetabolomeOut(BaseModel):
    id: str
    name: str
    n_metabolites: int


class MetabolomeUpdate(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


# ---------------- pH Functions ---------------- #
class PHDraftCreate(BaseModel):
    name: str
    baseValue: float = Field(7.0, ge=0.0, le=14.0)
    weights: Dict[str, float] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class PHDraftOut(BaseModel):
    id: str
    name: str
    baseValue: float
    n_weights: int


class PHDraftDetail(BaseModel):
    id: str
    name: str
    baseValue: float
    weights: Dict[str, float]


class PHDraftRename(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class PHFunctionCreate(BaseModel):
    name: str
    metabolome_id: str
    baseValue: float = Field(7.0, ge=0.0, le=14.0)
    weights: Dict[str, float] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class PHFunctionOut(BaseModel):
    id: str
    name: str
    metabolome_id: str
    metabolome_name: str
    n_metabolites: int


class PHFunctionDetail(BaseModel):
    id: str
    name: str
    metabolome_id: str
    baseValue: float
    weights: Dict[str, float]


class PHFunctionRename(BaseModel):
    name: str
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


# ---------------- Environments ---------------- #
class EnvironmentCreate(BaseModel):
    name: str
    ph_function_id: str
    temperature: float = Field(37.0, ge=0.0, le=100.0)
    stirring_rate: float = Field(1.0, ge=0.0, le=1.0)
    stirring_base_std: float = Field(0.1, ge=0.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class EnvironmentOut(BaseModel):
    id: str
    name: str
    ph_function_id: str
    ph_function_name: str
    metabolome_id: str
    metabolome_name: str
    temperature: float
    stirring_rate: float
    stirring_base_std: float


class EnvironmentDetail(EnvironmentOut):
    pass


class EnvironmentRename(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


# ---------------- Pulses ---------------- #
class PulseCreate(BaseModel):
    name: str
    t_start: float
    t_end: float
    n_steps: int
    vin: float = 0.0
    vout: float = 0.0
    qin: float = 0.0
    qout: float = 0.0
    environment_id: str
    feed_metabolome_instant_id: Optional[str] = None
    feed_metabolome_cont_id: Optional[str] = None
    # New: optional microbiome feeds (instantaneous and continuous)
    feed_microbiome_instant_id: Optional[str] = None
    feed_microbiome_cont_id: Optional[str] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class PulseOut(BaseModel):
    id: str
    name: str
    t_start: float
    t_end: float
    n_steps: int
    vin: float
    vout: float
    qin: float
    qout: float
    environment_id: str
    environment_name: str
    feed_metabolome_instant_id: Optional[str] = None
    feed_metabolome_instant_name: Optional[str] = None
    feed_metabolome_cont_id: Optional[str] = None
    feed_metabolome_cont_name: Optional[str] = None
    # New: microbiome feeds
    feed_microbiome_instant_id: Optional[str] = None
    feed_microbiome_instant_name: Optional[str] = None
    feed_microbiome_cont_id: Optional[str] = None
    feed_microbiome_cont_name: Optional[str] = None


class PulseDetail(PulseOut):
    pass


class PulseRename(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


# ---------------- Timelines ---------------- #
class TimelineCreate(BaseModel):
    name: str
    pulse_ids: List[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_tl_name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class TimelineOut(BaseModel):
    id: str
    name: str
    n_pulses: int
    t_start: float
    t_end: float


class TimelineDetail(BaseModel):
    id: str
    name: str
    pulses: List[PulseOut]


class TimelineRename(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_tl_name2(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


# ---------------- Simulations ---------------- #
class SimulationRunIn(BaseModel):
    name: str
    metabolome_id: str
    microbiome_id: str
    timeline_id: str
    volume: float = Field(1.0, ge=0.0)
    mode: str = Field("accurate", description="fast | balanced | accurate")


class SimulationResultOut(BaseModel):
    id: str
    name: str
    summary: Dict[str, float] = Field(default_factory=dict)
    plot: Dict[str, Any] = Field(default_factory=dict)


class SimulationListItem(BaseModel):
    id: str
    name: str

# ---------------- Bacteria / Microbiomes ---------------- #
class FeedingTermIn(BaseModel):
    id: str | None = None
    # Mapping metabolite name -> [yield, monodK]
    metDict: Dict[str, list[float]] = Field(default_factory=dict)

    @field_validator("metDict")
    @classmethod
    def validate_metdict(cls, m: Dict[str, list[float]]) -> Dict[str, list[float]]:
        out: Dict[str, list[float]] = {}
        for k, v in (m or {}).items():
            if not isinstance(v, (list, tuple)) or len(v) < 2:
                raise ValueError(f"metDict entry for {k!r} must be [yield, monodK]")
            try:
                y = float(v[0])
                K = float(v[1])
            except Exception as e:
                raise ValueError(f"invalid yield/monodK for {k!r}") from e
            out[str(k)] = [y, K]
        return out


class SubpopulationIn(BaseModel):
    name: str
    species: str
    count: float = 0.0
    mumax: float = 0.0
    feedingTerms: list[FeedingTermIn] = Field(default_factory=list)
    # pH
    pHopt: float = 7.0
    pH_sensitivity_left: float = 2.0
    pH_sensitivity_right: float = 2.0
    # Temperature
    Topt: float = 37.0
    tempSensitivity_left: float = 5.0
    tempSensitivity_right: float = 2.0
    # State/visuals
    state: str = "active"
    color: str = "#aaaaaa"

    @field_validator("name", "species")
    @classmethod
    def validate_nonempty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("value is required")
        if len(v) > 128:
            raise ValueError("value too long (max 128)")
        return v


class TransitionIn(BaseModel):
    target: str
    condition: str = ""
    rate: float = 0.0


class BacteriaIn(BaseModel):
    species: str
    color: str = "#54f542"
    subpopulations: list[SubpopulationIn] = Field(default_factory=list)
    # Map source subpopulation name -> list of [target, condition, rate]
    connections: Dict[str, list[TransitionIn]] = Field(default_factory=dict)

    @field_validator("species")
    @classmethod
    def validate_species(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("species is required")
        if len(v) > 128:
            raise ValueError("species too long (max 128)")
        return v


class BacteriaOut(BaseModel):
    id: str
    species: str
    n_subpops: int
    subpop_names: list[str] = Field(default_factory=list)


class BacteriaRename(BaseModel):
    species: str

    @field_validator("species")
    @classmethod
    def validate_species(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("species is required")
        if len(v) > 128:
            raise ValueError("species too long (max 128)")
        return v


class MicrobiomeCreate(BaseModel):
    name: str
    metabolome_id: str
    # Mapping species -> mapping subpop name -> count
    subpop_counts: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name2(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v


class MicrobiomeOut(BaseModel):
    id: str
    name: str
    n_species: int
    n_subpops: int


class MicrobiomeRename(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name3(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("name is required")
        if len(v) > 128:
            raise ValueError("name too long (max 128)")
        return v
