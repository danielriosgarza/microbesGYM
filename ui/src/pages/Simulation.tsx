import React from "react";
import { ApiStatus } from "../components/ApiStatus";
import CytoscapeComponent from "react-cytoscapejs";
import { listMetabolites, type MetaboliteOut } from "../lib/metabolites";
import { createMetabolome, listMetabolomes, deleteMetabolome, updateMetabolomeName, type MetabolomeOut as MetabolomeOutT } from "../lib/metabolomes";
import { listPHFunctions, createPHFunction, renamePHFunction, deletePHFunction, type PHFunctionOut } from "../lib/ph";
import { listPHDrafts, getPHDraft, deletePHDraft, type PHDraftOut } from "../lib/ph_drafts";
import { listEnvironments, createEnvironment, renameEnvironment, deleteEnvironment, type EnvironmentOut } from "../lib/env";
import { listPulses, createPulse, renamePulse, deletePulse, type PulseOut } from "../lib/pulses";
import { getCytoscape } from "../lib/viz";
import { listBacteria, type BacteriaOut } from "../lib/bacteria";
import { listMicrobiomes, createMicrobiome, renameMicrobiome, deleteMicrobiome, type MicrobiomeOut } from "../lib/microbiomes";
import { PulseTimeline } from "../components/PulseTimeline";
import { listTimelines, createTimeline, type TimelineOut } from "../lib/timelines";
import { runSimulation, type SimulationResultOut } from "../lib/simulations";
import reactorSvg from "../../static/img/reactor.svg";

export function Simulation() {
  const [count, setCount] = React.useState<number>(0);
  const [cySpec, setCySpec] = React.useState<any | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [metList, setMetList] = React.useState<MetaboliteOut[]>([]);
  const [initials, setInitials] = React.useState<Record<string, number>>({});
  const [metabolomeName, setMetabolomeName] = React.useState<string>("");
  const [creating, setCreating] = React.useState(false);
  const [createError, setCreateError] = React.useState<string | null>(null);
  const [createSuccess, setCreateSuccess] = React.useState<string | null>(null);
  const [phDrafts, setPhDrafts] = React.useState<Array<PHDraftOut & { metabolome_id?: string }>>([]);
  const [phFuncs, setPhFuncs] = React.useState<PHFunctionOut[]>([]);
  const [savingPhId, setSavingPhId] = React.useState<string | null>(null);
  const [envDrafts, setEnvDrafts] = React.useState<Array<{ id: string; name: string; ph_function_id?: string; temperature: number; stirring_rate: number; stirring_base_std: number }>>([]);
  const [envs, setEnvs] = React.useState<EnvironmentOut[]>([]);
  const [savingEnvId, setSavingEnvId] = React.useState<string | null>(null);
  const [pulseDrafts, setPulseDrafts] = React.useState<Array<{
    id: string;
    name: string;
    t_start: number;
    t_end: number;
    n_steps: number;
    vin: number;
    vout: number;
    qin: number;
    qout: number;
    environment_id?: string;
    feed_metabolome_instant_id?: string;
    feed_metabolome_cont_id?: string;
    feed_microbiome_instant_id?: string;
    feed_microbiome_cont_id?: string;
  }>>([]);
  const [pulses, setPulses] = React.useState<PulseOut[]>([]);
  const [savingPulseId, setSavingPulseId] = React.useState<string | null>(null);
  const [selPulseId, setSelPulseId] = React.useState<string | null>(null);
  const [selPulseKind, setSelPulseKind] = React.useState<"draft" | "saved" | null>(null);
  const [metabolomes, setMetabolomes] = React.useState<MetabolomeOutT[]>([]);
  const [bacteria, setBacteria] = React.useState<BacteriaOut[]>([]);
  const [subpopCounts, setSubpopCounts] = React.useState<Record<string, number>>({});
  const [microbiomeName, setMicrobiomeName] = React.useState<string>("");
  const [selectedMetabolomeId, setSelectedMetabolomeId] = React.useState<string>("");
  const [microbiomes, setMicrobiomes] = React.useState<MicrobiomeOut[]>([]);
  const [savingMicrobiome, setSavingMicrobiome] = React.useState<boolean>(false);
  const [editedNames, setEditedNames] = React.useState<Record<string, string>>({});
  const [savingId, setSavingId] = React.useState<string | null>(null);
  const [showPulseList, setShowPulseList] = React.useState<boolean>(true);
  const [timelines, setTimelines] = React.useState<Array<{ id: string; name: string; created_at: string; pulses: Array<{ id: string; name: string; t_start: number; t_end: number; n_steps: number; vin: number; vout: number; qin: number; qout: number; environment_id?: string; feed_metabolome_instant_id?: string; feed_metabolome_cont_id?: string }> }>>([]);
  // Backend timelines
  const [backendTimelines, setBackendTimelines] = React.useState<TimelineOut[]>([]);
  const [selectedTimelineId, setSelectedTimelineId] = React.useState<string>("");
  const [timelineName, setTimelineName] = React.useState<string>("");
  // Reactor inputs
  const [reactorName, setReactorName] = React.useState<string>("");
  const [reactorMode, setReactorMode] = React.useState<"fast" | "balanced" | "accurate">("accurate");
  const [reactorVolume, setReactorVolume] = React.useState<number>(1.0);
  const [selectedMicrobiomeId, setSelectedMicrobiomeId] = React.useState<string>("");
  // Track species that have been removed from microbiome interface
  const [removedSpecies, setRemovedSpecies] = React.useState<Set<string>>(new Set());

  // Load/save timelines from localStorage
  React.useEffect(() => {
    try {
      const raw = localStorage.getItem('mg-sim-timelines');
      const arr = raw ? (JSON.parse(raw) as any[]) : [];
      const mapped = arr.map((t) => ({
        id: String(t.id),
        name: String(t.name || ''),
        created_at: String(t.created_at || new Date().toISOString()),
        pulses: Array.isArray(t.pulses) ? t.pulses.map((p: any) => ({
          id: String(p.id), name: String(p.name || ''), t_start: Number(p.t_start)||0, t_end: Number(p.t_end)||0, n_steps: Number(p.n_steps)||1,
          vin: Number(p.vin)||0, vout: Number(p.vout)||0, qin: Number(p.qin)||0, qout: Number(p.qout)||0,
          environment_id: p.environment_id ? String(p.environment_id) : undefined,
          feed_metabolome_instant_id: p.feed_metabolome_instant_id ? String(p.feed_metabolome_instant_id) : undefined,
          feed_metabolome_cont_id: p.feed_metabolome_cont_id ? String(p.feed_metabolome_cont_id) : undefined,
        })) : [],
      }));
      setTimelines(mapped);
    } catch { setTimelines([]); }
  }, []);

  // Load/save removed species from localStorage
  React.useEffect(() => {
    try {
      const raw = localStorage.getItem('mg-sim-removed-species');
      const arr = raw ? (JSON.parse(raw) as string[]) : [];
      setRemovedSpecies(new Set(arr));
    } catch { setRemovedSpecies(new Set()); }
  }, []);

  React.useEffect(() => {
    try {
      localStorage.setItem('mg-sim-timelines', JSON.stringify(timelines));
    } catch {}
  }, [timelines]);

  // Load backend timelines
  const refreshBackendTimelines = React.useCallback(async () => {
    try {
      const list = await listTimelines();
      setBackendTimelines(list);
      if (selectedTimelineId && !list.find(t => t.id === selectedTimelineId)) setSelectedTimelineId("");
    } catch {
      setBackendTimelines([]);
      setSelectedTimelineId("");
    }
  }, [selectedTimelineId]);

  React.useEffect(() => { void refreshBackendTimelines(); }, [refreshBackendTimelines]);

  // Save removed species to localStorage when it changes
  React.useEffect(() => {
    try {
      localStorage.setItem('mg-sim-removed-species', JSON.stringify(Array.from(removedSpecies)));
    } catch {}
  }, [removedSpecies]);

  const nextPulseName = React.useCallback(() => {
    const names = [
      ...pulseDrafts.map((d) => d.name || ''),
      ...pulses.map((p) => p.name || ''),
    ];
    let maxN = 0;
    const re = /^pulse_(\d+)$/i;
    for (const n of names) {
      const m = re.exec(n.trim());
      if (m) { const v = Number(m[1]); if (v > maxN) maxN = v; }
    }
    return `pulse_${maxN + 1}`;
  }, [pulseDrafts, pulses]);

  const nextTimelineName = React.useCallback(() => {
    let maxN = 0;
    const re = /^timeline_(\d+)$/i;
    for (const t of timelines) {
      const m = re.exec((t.name || '').trim());
      if (m) { const v = Number(m[1]); if (v > maxN) maxN = v; }
    }
    return `timeline_${maxN + 1}`;
  }, [timelines]);

  const refreshMetabolomes = React.useCallback(async () => {
    try {
      const list = await listMetabolomes();
      setMetabolomes(list);
    } catch {
      setMetabolomes([]);
    }
  }, []);

  React.useEffect(() => {
    void refreshMetabolomes();
  }, [refreshMetabolomes]);

  // Apply initials pushed from Build tab (metabolites:initials)
  React.useEffect(() => {
    const onInit = (e: Event) => {
      try {
        const detail = (e as CustomEvent).detail as { initials?: Record<string, number> } | undefined;
        if (detail && detail.initials && typeof detail.initials === 'object') {
          setInitials({ ...detail.initials });
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Failed to apply pushed initials', err);
      }
    };
    window.addEventListener('metabolites:initials', onInit as EventListener);
    return () => window.removeEventListener('metabolites:initials', onInit as EventListener);
  }, []);

  // Load saved pH functions, environments, pulses
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [funcs, envList, pulseList] = await Promise.all([listPHFunctions(), listEnvironments(), listPulses()]);
        if (!cancelled) { setPhFuncs(funcs); setEnvs(envList); setPulses(pulseList); }
      } catch {
        if (!cancelled) { setPhFuncs([]); setEnvs([]); setPulses([]); }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Load bacteria and microbiomes and update subpop count keys
  const refreshBacteria = React.useCallback(async () => {
    console.log('refreshBacteria called - fetching bacteria list...');
    try {
      const list = await listBacteria();
      console.log('Bacteria list received:', list);
      setBacteria(list);
      
      // Clear removedSpecies for any bacteria that are now present in the list
      setRemovedSpecies(prev => {
        const newRemoved = new Set(prev);
        for (const b of list) {
          newRemoved.delete(b.species);
        }
        return newRemoved;
      });
      
      setSubpopCounts((prev) => {
        const next = { ...prev };
        const keys = new Set(Object.keys(next));
        for (const b of list) {
          for (const sp of (b.subpop_names || [])) {
            const k = `${b.species}::${sp}`;
            if (!(k in next)) next[k] = 0;
          }
        }
        // Remove keys for species no longer present
        for (const k of Object.keys(next)) {
          const [spec, sp] = k.split("::");
          const found = list.find((b) => b.species === spec && (b.subpop_names || []).includes(sp));
          if (!found) delete next[k];
        }

        return next;
      });
    } catch (error) {
      console.error('Error fetching bacteria list:', error);
      setBacteria([]);
    }
    try {
      const mic = await listMicrobiomes();
      console.log('Microbiomes list received:', mic);
      setMicrobiomes(mic);
    } catch (error) {
      console.error('Error fetching microbiomes list:', error);
      setMicrobiomes([]);
    }
  }, []);

  React.useEffect(() => {
    // Check if bacteria were updated while this component was not active
    const bacteriaUpdated = localStorage.getItem('bacteria_updated');
    if (bacteriaUpdated) {
      console.log('Bacteria were updated while Simulation tab was inactive, refreshing...');
      localStorage.removeItem('bacteria_updated');
    }
    
    void refreshBacteria();
    const onBac = () => { 
      console.log('bacteria:changed event received, refreshing bacteria list');
      void refreshBacteria(); 
    };
    window.addEventListener('bacteria:changed', onBac);
    return () => window.removeEventListener('bacteria:changed', onBac);
  }, [refreshBacteria]);

  // Add a global event listener that persists even when component unmounts
  React.useEffect(() => {
    const globalOnBac = () => {
      console.log('Global bacteria:changed event received');
      // Store a flag in localStorage to indicate bacteria were updated
      localStorage.setItem('bacteria_updated', Date.now().toString());
    };
    window.addEventListener('bacteria:changed', globalOnBac);
    return () => window.removeEventListener('bacteria:changed', globalOnBac);
  }, []);

  // Also refresh bacteria when the component mounts or becomes visible
  React.useEffect(() => {
    const handleVisibilityChange = () => {
      if (!document.hidden) {
        console.log('Simulation tab became visible, refreshing bacteria list');
        void refreshBacteria();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [refreshBacteria]);

  // Add a periodic check for bacteria updates when tab is active
  React.useEffect(() => {
    const interval = setInterval(() => {
      const bacteriaUpdated = localStorage.getItem('bacteria_updated');
      if (bacteriaUpdated) {
        console.log('Periodic check: bacteria were updated, refreshing...');
        localStorage.removeItem('bacteria_updated');
        void refreshBacteria();
      }
    }, 1000); // Check every second

    return () => clearInterval(interval);
  }, [refreshBacteria]);

  // Load pH drafts from backend
  const loadDrafts = React.useCallback(async () => {
    try {
      const list = await listPHDrafts();
      setPhDrafts(list);
    } catch { setPhDrafts([]); }
  }, []);

  React.useEffect(() => {
    void loadDrafts();
    const onDraft = () => { void loadDrafts(); };
    window.addEventListener('ph:draftSaved', onDraft);
    return () => { window.removeEventListener('ph:draftSaved', onDraft); };
  }, [loadDrafts]);

  // Respond to global data reset (e.g., Build tab "Delete all")
  React.useEffect(() => {
    const onReset = async () => {
      try {
        // Hide any currently shown Cytoscape graph and clear errors
        setCySpec(null);
        setError(null);
        const [funcs, envList, pulseList] = await Promise.all([
          listPHFunctions().catch(() => []),
          listEnvironments().catch(() => []),
          listPulses().catch(() => []),
        ]);
        setPhFuncs(funcs);
        setEnvs(envList);
        setPulses(pulseList);
        // Also refresh metabolomes and backend timelines to reflect removals
        void refreshMetabolomes();
        void refreshBackendTimelines();
        // Clear local drafts/snapshots
        try { setTimelines([]); } catch {}
        try { loadDrafts(); } catch {}
      } catch {
        setCySpec(null);
        setError(null);
        setPhFuncs([]);
        setEnvs([]);
        setPulses([]);
        void refreshMetabolomes();
        void refreshBackendTimelines();
        try { setTimelines([]); } catch {}
        try { loadDrafts(); } catch {}
      }
    };
    window.addEventListener('mg:dataReset', onReset);
    return () => window.removeEventListener('mg:dataReset', onReset);
  }, [refreshMetabolomes, loadDrafts]);

  // Keep the metabolite count fresh without requiring a page refresh
  React.useEffect(() => {
    let cancelled = false;
    const refreshCount = async () => {
      try {
        const mets = await listMetabolites();
        if (cancelled) return;
        setCount(mets.length);
        setMetList(mets);
        // Initialize initials if empty or metabolites changed
        setInitials((prev) => {
          if (Object.keys(prev).length === 0) {
            const init: Record<string, number> = {};
            for (const m of mets) init[m.name] = m.concentration ?? 0;
            return init;
          }
          // Ensure all current metabolites exist in map
          const next: Record<string, number> = { ...prev };
          for (const m of mets) if (!(m.name in next)) next[m.name] = m.concentration ?? 0;
          // Drop removed ones
          for (const k of Object.keys(next)) if (!mets.find((m) => m.name === k)) delete next[k];
          return next;
        });
      } catch {
        if (!cancelled) setCount(0);
      }
    };
    // initial fetch + periodic refresh
    refreshCount();
    const id = setInterval(refreshCount, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const handleInspect = async () => {
    setLoading(true);
    setError(null);
    try {
      // Also refresh the count when inspecting
      try {
        const mets = await listMetabolites();
        setCount(mets.length);
      } catch {}
      const spec = await getCytoscape();
      setCySpec(spec);
    } catch (e) {
      setError("Failed to build Cytoscape spec");
    } finally {
      setLoading(false);
    }
  };

  const elementsArray = cySpec
    ? (Array.isArray(cySpec.elements)
        ? cySpec.elements
        : [
            ...((cySpec.elements && cySpec.elements.nodes) || []),
            ...((cySpec.elements && cySpec.elements.edges) || []),
          ])
    : [];

  return (
    <div className="grid" id="panel-simulation">
      {/* Reactor builder (top) */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Reactor</h2>
        <div className="row" style={{ gap: 16, alignItems: 'stretch', marginTop: 8 }}>
          <div style={{ width: 260, maxWidth: '30%' }}>
            <img src={reactorSvg} alt="Reactor schematic" style={{ width: '100%', height: 'auto', display: 'block' }} />
          </div>
          <div className="form" style={{ flex: 1 }}>
          <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
            <label className="label">Name
              <input className="input" placeholder="reactor name" value={reactorName} onChange={(e) => setReactorName(e.target.value)} />
            </label>
            <label className="label">Mode
              <select className="input" value={reactorMode} onChange={(e) => setReactorMode(e.target.value as any)}>
                <option value="fast">Fast</option>
                <option value="balanced">Moderate</option>
                <option value="accurate">Accurate</option>
              </select>
            </label>
            <label className="label">Volume
              <input className="input" type="number" min={0} step={0.1} value={reactorVolume} onChange={(e) => setReactorVolume(Number(e.target.value))} />
            </label>
            <div />
          </div>
          <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
            <label className="label">Metabolome
              <select className="input" value={selectedMetabolomeId} onChange={(e) => setSelectedMetabolomeId(e.target.value)}>
                <option value="">Select metabolome</option>
                {metabolomes.map((m) => (<option key={m.id} value={m.id}>{m.name} ({m.n_metabolites})</option>))}
              </select>
            </label>
            {/* removed duplicate placeholder Microbiome labels */}
          </div>
          <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
            <label className="label">Microbiome
              <select className="input" id="reactor-micro" value={selectedMicrobiomeId} onChange={(e) => setSelectedMicrobiomeId(e.target.value)}>
                <option value="">Select microbiome</option>
                {microbiomes.map((m) => (<option key={m.id} value={m.id}>{m.name} ({m.n_species} sp)</option>))}
              </select>
            </label>
            <label className="label">Timeline
              <select className="input" value={selectedTimelineId} onChange={(e) => setSelectedTimelineId(e.target.value)}>
                <option value="">Select timeline</option>
                {backendTimelines.map((t) => (<option key={t.id} value={t.id}>{t.name} ({t.n_pulses} pulses)</option>))}
              </select>
            </label>
            <div />
          </div>
          <div className="row" style={{ justifyContent: 'flex-end', gap: 8 }}>
            <button className="btn primary" disabled={!reactorName.trim() || !selectedMetabolomeId || !selectedMicrobiomeId || !selectedTimelineId} onClick={async () => {
              try {
                setLoading(true);
                const res: SimulationResultOut = await runSimulation({ name: reactorName.trim(), metabolome_id: selectedMetabolomeId, microbiome_id: selectedMicrobiomeId, timeline_id: selectedTimelineId, volume: reactorVolume, mode: reactorMode });
                try { window.dispatchEvent(new CustomEvent('simulation:completed', { detail: { id: res.id } })); } catch {}
                alert('Simulation completed');
              } catch (e) { alert('Failed to run simulation'); } finally { setLoading(false); }
            }}>Simulate</button>
          </div>
          </div>
        </div>
      </div>
      <div className="card" style={{ gridColumn: "span 12" }}>
        <div className="row space-between center">
          <ApiStatus />
          <div className="muted">{count > 0 ? `Valid model with ${count} metabolite${count>1?'s':''}` : "No saved metabolites"}</div>
        </div>
      </div>

      {/* Initial Conditions builder */}
      {metList.length > 0 && (
        <div className="card" style={{ gridColumn: "span 12" }}>
          <h2>Initial Conditions: metabolites</h2>
          <div className="form" style={{ marginTop: 8 }}>
            <div className="form-row">
              <label className="label" htmlFor="metName">Metabolome name</label>
              <input
                id="metName"
                className="input"
                placeholder="e.g., baseline"
                value={metabolomeName}
                onChange={(e) => setMetabolomeName(e.target.value)}
              />
            </div>
            <div className="form-row">
              <label className="label">Concentrations (mM)</label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
                {metList.map((m) => (
                  <div key={m.id} className="inline" style={{ gap: 6 }}>
                    <span className="chip" style={{ minWidth: 0 }}>
                      <span style={{ width: 10, height: 10, borderRadius: 999, background: m.color, display: 'inline-block', marginRight: 6 }} />
                      {m.name}
                    </span>
                    <input
                      type="number"
                      className="input"
                      min={0}
                      step={0.1}
                      value={Number.isFinite(initials[m.name]) ? initials[m.name] : 0}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        setInitials((prev) => ({ ...prev, [m.name]: v }));
                      }}
                    />
                    <span className="unit">mM</span>
                  </div>
                ))}
              </div>
            </div>
            {createError && <div className="hint error">{createError}</div>}
            {createSuccess && <div className="hint">{createSuccess}</div>}
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button
                className="btn primary"
                disabled={creating || metList.length === 0}
                onClick={async () => {
                  try {
                    setCreating(true);
                    setCreateError(null);
                    setCreateSuccess(null);
                    const name = metabolomeName.trim() || `metabolome-${new Date().toISOString().slice(0,10)}`;
                    const payload = { name, concentrations: initials };
                    await createMetabolome(payload);
                    setMetabolomeName("");
                    setCreateSuccess("Metabolome created");
                    // Update local list and notify other tabs
                    await refreshMetabolomes();
                    window.dispatchEvent(new CustomEvent("metabolomes:changed"));
                  } catch (e) {
                    const msg = e instanceof Error ? e.message : "Failed to create metabolome";
                    setCreateError(msg);
                  } finally {
                    setCreating(false);
                  }
                }}
              >
                {creating ? 'Creating…' : 'Create metabolome'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Subpopulation counts and microbiome creation */}
      {bacteria.length > 0 && (
        <div className="card" style={{ gridColumn: "span 12" }}>
          <div className="row space-between center">
            <h2>Initial Conditions: subpopulations</h2>
            <button 
              className="btn" 
              onClick={() => {
                console.log('Manual refresh of bacteria list');
                void refreshBacteria();
              }}
              title="Refresh bacteria list"
            >
              Refresh
            </button>
          </div>
          <div className="form" style={{ marginTop: 8 }}>
            <div className="form-row">
              <label className="label">Subpopulation counts</label>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
                {bacteria
                  .filter(b => !removedSpecies.has(b.species)) // Filter out removed species
                  .flatMap((b) => (b.subpop_names || []).map((sp) => {
                    const key = `${b.species}::${sp}`;
                    return (
                    <div key={key} className="inline" style={{ gap: 6 }}>
                      <span className="chip" style={{ minWidth: 0 }}>{b.species} – {sp}</span>
                      <input
                        type="number"
                        className="input"
                        min={0}
                        step={0.001}
                        value={Number.isFinite(subpopCounts[key]) ? subpopCounts[key] : 0}
                        onChange={(e) => {
                          const v = Number(e.target.value);
                          setSubpopCounts((prev) => ({ ...prev, [key]: v }));
                        }}
                      />
                      <button 
                        className="btn" 
                        style={{ padding: '4px 8px', fontSize: '12px' }}
                        onClick={() => {
                          // Remove this species from microbiome interface
                          setRemovedSpecies(prev => new Set([...prev, b.species]));
                          // Clear subpop counts for this species
                          setSubpopCounts(prev => {
                            const next = { ...prev };
                            Object.keys(next).forEach(key => {
                              if (key.startsWith(`${b.species}::`)) {
                                delete next[key];
                              }
                            });
                            return next;
                          });
                        }}
                        title={`Remove ${b.species} from microbiome`}
                      >
                        ×
                      </button>
                    </div>
                  );
                }))}
              </div>
            </div>
            

            

            <div className="form-row">
              <label className="label" htmlFor="mic-name">Microbiome name</label>
              <input id="mic-name" className="input" value={microbiomeName} onChange={(e) => setMicrobiomeName(e.target.value)} />
            </div>
            <div className="form-row">
              <label className="label" htmlFor="mic-met">Bind to metabolome</label>
              <select id="mic-met" className="input" value={selectedMetabolomeId} onChange={(e) => setSelectedMetabolomeId(e.target.value)}>
                <option value="">Select metabolome</option>
                {metabolomes.map((m) => (
                  <option key={m.id} value={m.id}>{m.name} ({m.n_metabolites})</option>
                ))}
              </select>
            </div>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn primary" disabled={savingMicrobiome || !selectedMetabolomeId} onClick={async () => {
                try {
                  setSavingMicrobiome(true);
                  const name = (microbiomeName || `microbiome-${new Date().toISOString().slice(0,10)}`).trim();
                  // Build nested subpop_counts: { species: { subpop: count } }
                  const nested: Record<string, Record<string, number>> = {};
                  for (const [key, val] of Object.entries(subpopCounts)) {
                    const [spec, sp] = key.split('::');
                    if (!nested[spec]) nested[spec] = {};
                    nested[spec][sp] = Number(val || 0) || 0;
                  }
                  const payload = { name, metabolome_id: selectedMetabolomeId, subpop_counts: nested };
                  await createMicrobiome(payload);
                  setMicrobiomeName("");
                  await refreshBacteria();
                  try { window.dispatchEvent(new CustomEvent('microbiomes:changed')); } catch {}
                } catch (e) {
                  const msg = e instanceof Error ? e.message : String(e);
                  alert(`Failed to create microbiome: ${msg}`);
                } finally {
                  setSavingMicrobiome(false);
                }
              }}>{savingMicrobiome ? 'Saving…' : 'Create microbiome'}</button>
            </div>
          </div>
        </div>
      )}

      {/* Metabolomes manager (moved above Microbiomes) */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Metabolomes</h2>
        {metabolomes.length === 0 ? (
          <p className="muted">No metabolomes yet.</p>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {metabolomes.map((m) => {
              const current = editedNames[m.id] ?? m.name;
              const dirty = current.trim() !== m.name;
              return (
                <div key={m.id} className="row" style={{ gap: 8, alignItems: 'center' }}>
                  <input
                    className="input"
                    value={current}
                    onChange={(e) => setEditedNames((prev) => ({ ...prev, [m.id]: e.target.value }))}
                    style={{ maxWidth: 320 }}
                  />
                  <span className="muted">{m.n_metabolites} metabolites</span>
                  <div style={{ flex: 1 }} />
                  <button
                    className="btn primary"
                    onClick={async () => {
                      try {
                        const spec = await getCytoscape(undefined, { metabolome_id: m.id });
                        setCySpec(spec);
                        setError(null);
                      } catch {
                        alert('Failed to inspect');
                      }
                    }}
                  >
                    Inspect
                  </button>
                  <button
                    className="btn"
                    disabled={!dirty || !current.trim() || savingId === m.id}
                    onClick={async () => {
                      try {
                        setSavingId(m.id);
                        const newName = current.trim();
                        await updateMetabolomeName(m.id, newName);
                        await refreshMetabolomes();
                        window.dispatchEvent(new CustomEvent("metabolomes:changed"));
                      } catch (e) {
                        alert("Failed to rename metabolome");
                      } finally {
                        setSavingId(null);
                      }
                    }}
                  >
                    {savingId === m.id ? 'Saving.' : 'Save name'}
                  </button>
                  <button
                    className="btn"
                    disabled={savingId === m.id}
                    onClick={async () => {
                      const ok = window.confirm(`Delete metabolome "${m.name}"?`);
                      if (!ok) return;
                      try {
                        setSavingId(m.id);
                        await deleteMetabolome(m.id);
                        await refreshMetabolomes();
                        window.dispatchEvent(new CustomEvent("metabolomes:changed"));
                      } catch (e) {
                        alert("Failed to delete metabolome");
                      } finally {
                        setSavingId(null);
                      }
                    }}
                  >
                    Delete
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Microbiomes list */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Microbiomes</h2>
        {microbiomes.length === 0 ? (
          <p className="muted">No microbiomes yet.</p>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {microbiomes.map((m) => (
              <div key={m.id} className="row" style={{ gap: 8, alignItems: 'center' }}>
                <input className="input" value={m.name} onChange={(e) => {
                  const name = e.target.value;
                  setMicrobiomes((prev) => prev.map((x) => x.id === m.id ? { ...x, name } : x));
                }} style={{ maxWidth: 260 }} />
                <span className="chip">{m.n_species} species</span>
                <span className="chip">{m.n_subpops} subpops</span>
                <button className="btn" onClick={async () => { try { await renameMicrobiome(m.id, (m.name || '').trim() || `microbiome-${new Date().toISOString().slice(0,10)}`); await refreshBacteria(); try { window.dispatchEvent(new CustomEvent('microbiomes:changed')); } catch {} } catch { alert('Failed to rename'); } }}>Save name</button>
                <button className="btn" onClick={async () => { try { await deleteMicrobiome(m.id); await refreshBacteria(); try { window.dispatchEvent(new CustomEvent('microbiomes:changed')); } catch {} } catch { alert('Failed to delete'); } }}>Delete</button>
                <button className="btn primary" onClick={async () => { try { const spec = await getCytoscape(undefined, { microbiome_id: m.id }); setCySpec(spec); } catch { alert('Failed to inspect'); } }}>Inspect</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Metabolomes manager */}
      <div className="card" style={{ gridColumn: "span 12", display: 'none' }}>
        <h2>Metabolomes</h2>
        {metabolomes.length === 0 ? (
          <p className="muted">No metabolomes yet.</p>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            {metabolomes.map((m) => {
              const current = editedNames[m.id] ?? m.name;
              const dirty = current.trim() !== m.name;
              return (
                <div key={m.id} className="row" style={{ gap: 8, alignItems: 'center' }}>
                  <input
                    className="input"
                    value={current}
                    onChange={(e) => setEditedNames((prev) => ({ ...prev, [m.id]: e.target.value }))}
                    style={{ maxWidth: 320 }}
                  />
                  <span className="muted">{m.n_metabolites} metabolites</span>
                  <div style={{ flex: 1 }} />
                  <button
                    className="btn"
                    disabled={!dirty || !current.trim() || savingId === m.id}
                    onClick={async () => {
                      try {
                        setSavingId(m.id);
                        const newName = current.trim();
                        await updateMetabolomeName(m.id, newName);
                        await refreshMetabolomes();
                        window.dispatchEvent(new CustomEvent("metabolomes:changed"));
                      } catch (e) {
                        alert("Failed to rename metabolome");
                      } finally {
                        setSavingId(null);
                      }
                    }}
                  >
                    {savingId === m.id ? 'Saving…' : 'Save name'}
                  </button>
                  <button
                    className="btn"
                    disabled={savingId === m.id}
                    onClick={async () => {
                      const ok = window.confirm(`Delete metabolome "${m.name}"?`);
                      if (!ok) return;
                      try {
                        setSavingId(m.id);
                        await deleteMetabolome(m.id);
                        await refreshMetabolomes();
                        window.dispatchEvent(new CustomEvent("metabolomes:changed"));
                      } catch (e) {
                        alert("Failed to delete metabolome");
                      } finally {
                        setSavingId(null);
                      }
                    }}
                  >
                    Delete
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {(phDrafts.length > 0 || phFuncs.length > 0) && (
        <div className="card" style={{ gridColumn: "span 12" }}>
          <h2>pH Functions</h2>
          {phDrafts.map((d) => (
            <div key={d.id} className="row" style={{ gap: 8, alignItems: 'center', marginTop: 6 }}>
              <span aria-label="unsaved" title="Unsaved draft" style={{ width: 8, height: 8, borderRadius: 999, background: '#f87171', display: 'inline-block' }} />
              <span className="chip" style={{ minWidth: 0 }}>{d.name}</span>
              <select
                className="input"
                value={d.metabolome_id || ''}
                onChange={(e) => {
                  const metabolome_id = e.target.value || undefined;
                  setPhDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, metabolome_id } : x));
                }}
                style={{ minWidth: 220 }}
              >
                <option value="">Select metabolome</option>
                {metabolomes.map((m) => (
                  <option key={m.id} value={m.id}>{m.name} ({m.n_metabolites})</option>
                ))}
              </select>
              <button
                className="btn primary"
                disabled={!d.metabolome_id || savingPhId === d.id}
                onClick={async () => {
                  if (!d.metabolome_id) return;
                  try {
                    setSavingPhId(d.id);
                    const detail = await getPHDraft(d.id);
                    const payload = { name: (d.name || `ph-${new Date().toISOString().slice(0,10)}`), metabolome_id: d.metabolome_id, baseValue: detail.baseValue, weights: detail.weights };
                    const saved = await createPHFunction(payload);
                    setPhFuncs((prev) => [saved, ...prev]);
                    await deletePHDraft(d.id).catch(() => {});
                    await loadDrafts();
                  } catch {
                    alert('Failed to save pH function');
                  } finally { setSavingPhId(null); }
                }}
              >
                {savingPhId === d.id ? 'Saving…' : 'Save'}
              </button>
              <button
                className="btn"
                onClick={() => {
                  deletePHDraft(d.id).catch(() => {});
                  setPhDrafts((prev) => prev.filter((x) => x.id !== d.id));
                }}
              >
                Delete
              </button>
            </div>
          ))}

          {phFuncs.map((f) => (
            <div key={f.id} className="row" style={{ gap: 8, alignItems: 'center', marginTop: 6 }}>
              <span aria-label="saved" title="Saved" style={{ width: 8, height: 8, borderRadius: 999, background: '#22d3ee', display: 'inline-block' }} />
              <input
                className="input"
                value={f.name}
                onChange={(e) => setPhFuncs((prev) => prev.map((x) => x.id === f.id ? { ...x, name: e.target.value } : x))}
                style={{ maxWidth: 260 }}
              />
              <span className="chip" style={{ minWidth: 0 }}>{f.metabolome_name}</span>
              <div style={{ flex: 1 }} />
              <button
                className="btn"
                disabled={savingPhId === f.id}
                onClick={async () => {
                  try {
                    setSavingPhId(f.id);
                    const updated = await renamePHFunction(f.id, f.name);
                    setPhFuncs((prev) => prev.map((x) => x.id === f.id ? { ...x, name: updated.name } : x));
                  } catch { alert('Failed to rename pH function'); }
                  finally { setSavingPhId(null); }
                }}
              >
                Save name
              </button>
              <button
                className="btn"
                disabled={savingPhId === f.id}
                onClick={async () => {
                  const ok = window.confirm(`Delete pH function "${f.name}"?`);
                  if (!ok) return;
                  try {
                    setSavingPhId(f.id);
                    await deletePHFunction(f.id);
                    setPhFuncs((prev) => prev.filter((x) => x.id !== f.id));
                  } catch { alert('Failed to delete pH function'); }
                  finally { setSavingPhId(null); }
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Environments */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <div className="row space-between center">
          <h2>Environments</h2>
          <button className="btn" onClick={() => setEnvDrafts((prev) => [
            { id: `env-draft-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`, name: '', temperature: 37, stirring_rate: 1, stirring_base_std: 0.1 },
            ...prev,
          ])}>New environment</button>
        </div>
        {envDrafts.length > 0 && (
          <div className="row" style={{ gap: 8, alignItems: 'center', marginTop: 8, fontSize: 12 }}>
            <span style={{ width: 8 }} />
            <span className="muted" style={{ width: 240 }}>Name</span>
            <span className="muted" style={{ width: 240 }}>pH function</span>
            <span className="muted" style={{ width: 120 }}>Temperature (°C)</span>
            <span className="muted" style={{ width: 120 }}>Stirring rate</span>
            <span className="muted" style={{ width: 120 }}>Base std</span>
            <span style={{ flex: 1 }} />
          </div>
        )}
        {envDrafts.map((d) => (
          <div key={d.id} className="row" style={{ gap: 8, alignItems: 'center', marginTop: 6 }}>
            <span aria-label="unsaved" title="Unsaved draft" style={{ width: 8, height: 8, borderRadius: 999, background: '#f87171', display: 'inline-block' }} />
            <input className="input" placeholder="Environment name" value={d.name} onChange={(e) => setEnvDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, name: e.target.value } : x))} style={{ width: 240 }} />
            <select className="input" value={d.ph_function_id || ''} onChange={(e) => setEnvDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, ph_function_id: e.target.value || undefined } : x))} style={{ width: 240 }}>
              <option value="">Select pH function</option>
              {phFuncs.map((f) => (<option key={f.id} value={f.id}>{f.name} · {f.metabolome_name}</option>))}
            </select>
            <input className="input" type="number" step={0.1} min={0} max={100} value={d.temperature} onChange={(e) => setEnvDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, temperature: Number(e.target.value) } : x))} placeholder="Temp (°C)" style={{ width: 120 }} />
            <input className="input" type="number" step={0.01} min={0} max={1} value={d.stirring_rate} onChange={(e) => setEnvDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, stirring_rate: Number(e.target.value) } : x))} placeholder="Rate (0-1)" style={{ width: 120 }} />
            <input className="input" type="number" step={0.01} min={0} value={d.stirring_base_std} onChange={(e) => setEnvDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, stirring_base_std: Number(e.target.value) } : x))} placeholder="Base std" style={{ width: 120 }} />
            <button className="btn primary" disabled={!d.name.trim() || !d.ph_function_id || savingEnvId === d.id} onClick={async () => {
              try {
                setSavingEnvId(d.id);
                const payload = { name: d.name.trim(), ph_function_id: d.ph_function_id!, temperature: d.temperature, stirring_rate: d.stirring_rate, stirring_base_std: d.stirring_base_std };
                const saved = await createEnvironment(payload);
                setEnvs((prev) => [saved, ...prev]);
                setEnvDrafts((prev) => prev.filter((x) => x.id !== d.id));
              } catch { alert('Failed to save environment'); } finally { setSavingEnvId(null); }
            }}>{savingEnvId === d.id ? 'Saving…' : 'Save'}</button>
            <button className="btn" onClick={() => setEnvDrafts((prev) => prev.filter((x) => x.id !== d.id))}>Delete</button>
          </div>
        ))}

        {envs.map((e) => (
          <div key={e.id} className="row" style={{ gap: 8, alignItems: 'center', marginTop: 6 }}>
            <span aria-label="saved" title="Saved" style={{ width: 8, height: 8, borderRadius: 999, background: '#22d3ee', display: 'inline-block' }} />
            <input className="input" value={e.name} onChange={(ev) => setEnvs((prev) => prev.map((x) => x.id === e.id ? { ...x, name: ev.target.value } : x))} style={{ maxWidth: 240 }} />
            <span className="chip" style={{ minWidth: 0 }}>{e.ph_function_name} · {e.metabolome_name}</span>
            <span className="muted">{e.temperature} C, rate {e.stirring_rate}, std {e.stirring_base_std}</span>
            <div style={{ flex: 1 }} />
            <button className="btn" disabled={savingEnvId === e.id} onClick={async () => {
              try { setSavingEnvId(e.id); const up = await renameEnvironment(e.id, e.name); setEnvs((prev) => prev.map((x) => x.id === e.id ? { ...x, name: up.name } : x)); } catch { alert('Failed to rename environment'); } finally { setSavingEnvId(null); }
            }}>Save name</button>
            <button className="btn" disabled={savingEnvId === e.id} onClick={async () => {
              const ok = window.confirm(`Delete environment "${e.name}"?`); if (!ok) return; try { setSavingEnvId(e.id); await deleteEnvironment(e.id); setEnvs((prev) => prev.filter((x) => x.id !== e.id)); } catch { alert('Failed to delete environment'); } finally { setSavingEnvId(null); }
            }}>Delete</button>
            
          </div>
        ))}
      </div>

      {/* Pulses timeline + inspector */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Pulses</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 12, alignItems: 'start' }}>
          <div style={{ minWidth: 0 }}>
            <div className="row space-between center" style={{ marginBottom: 8 }}>
              <div className="row" style={{ gap: 8, alignItems: 'center' }}>
                <span className="muted">Timeline</span>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <label className="label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  Name
                  <input className="input" placeholder="timeline name" value={timelineName} onChange={(e) => setTimelineName(e.target.value)} style={{ width: 220 }} />
                </label>
                <button className="btn" onClick={async () => {
                  try {
                    // Create a backend timeline from currently saved pulses only
                    const ids = pulses.map((p) => p.id);
                    if (ids.length === 0) { alert('No saved pulses to include in timeline.'); return; }
                    const name = (timelineName || nextTimelineName()).trim();
                    const savedTl = await createTimeline({ name, pulse_ids: ids });
                    await refreshBackendTimelines();
                    setSelectedTimelineId(savedTl.id);
                    setTimelineName("");
                  } catch (e) {
                    alert('Failed to save timeline');
                  }
                }}>Save timeline</button>
                <button className="btn" onClick={() => {
                  const lastSaved = pulses.length > 0 ? Math.max(...pulses.map((p) => p.t_end)) : 0;
                  const lastDraft = pulseDrafts.length > 0 ? Math.max(...pulseDrafts.map((p) => p.t_end)) : 0;
                  const lastEnd = Math.max(lastSaved, lastDraft);
                  const id = `pulse-draft-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
                  const name = nextPulseName();
                  setPulseDrafts((prev) => [{ id, name, t_start: lastEnd, t_end: lastEnd + 1, n_steps: 60, vin: 0, vout: 0, qin: 0, qout: 0 }, ...prev]);
                }}>+ New pulse</button>
              </div>
            </div>
            <PulseTimeline
              saved={pulses.map((p) => ({ id: p.id, name: p.name, t_start: p.t_start, t_end: p.t_end, n_steps: p.n_steps }))}
              drafts={pulseDrafts.map((d) => ({ id: d.id, name: d.name, t_start: d.t_start, t_end: d.t_end, n_steps: d.n_steps }))}
              onSelect={(id, kind) => { setSelPulseId(id); setSelPulseKind(kind); }}
              onDraftChange={(id, patch) => setPulseDrafts((prev) => prev.map((x) => (x.id === id ? { ...x, ...patch } : x)))}
            />
          </div>

          <div className="form">
            {!selPulseId || !selPulseKind ? (
              <p className="muted">Select a pulse on the timeline to edit its properties.</p>
            ) : selPulseKind === 'draft' ? (
            (() => {
              const d = pulseDrafts.find((x) => x.id === selPulseId);
              if (!d) return <p className="muted">Draft not found.</p>;
              return (
                <>
                  <div className="form-row">
                    <label className="label">Name</label>
                    <input className="input" value={d.name} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, name: e.target.value } : x))} />
                  </div>
                  <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
                    <label className="label">Start (h)
                      <input className="input" type="number" step={0.25} min={0} value={d.t_start} onChange={(e) => {
                        const v = Number(e.target.value);
                        const dur = Math.max(0.25, d.t_end - v);
                        const steps = Math.max(1, Math.round(dur * 60));
                        setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, t_start: v, n_steps: steps } : x));
                      }} />
                    </label>
                    <label className="label">End (h)
                      <input className="input" type="number" step={0.25} min={d.t_start + 0.25} value={d.t_end} onChange={(e) => {
                        const v = Number(e.target.value);
                        const dur = Math.max(0.25, v - d.t_start);
                        const steps = Math.max(1, Math.round(dur * 60));
                        setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, t_end: v, n_steps: steps } : x));
                      }} />
                    </label>
                    <label className="label">Steps
                      <input className="input" type="number" min={1} step={1} value={d.n_steps} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, n_steps: Number(e.target.value) } : x))} />
                    </label>
                    <div />
                  </div>
                  <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 8 }}>
                    <label className="label">vin
                      <input className="input" type="number" min={0} step={0.1} value={d.vin} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, vin: Number(e.target.value) } : x))} />
                    </label>
                    <label className="label">vout
                      <input className="input" type="number" min={0} step={0.1} value={d.vout} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, vout: Number(e.target.value) } : x))} />
                    </label>
                    <label className="label">qin
                      <input className="input" type="number" min={0} step={0.01} value={d.qin} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, qin: Number(e.target.value) } : x))} />
                    </label>
                    <label className="label">qout
                      <input className="input" type="number" min={0} step={0.01} value={d.qout} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, qout: Number(e.target.value) } : x))} />
                    </label>
                  </div>
                  <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 8 }}>
                    <label className="label">Environment
                      <select className="input" value={d.environment_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, environment_id: e.target.value || undefined } : x))}>
                        <option value="">Select environment</option>
                        {envs.map((e) => (<option key={e.id} value={e.id}>{e.name} · {e.metabolome_name}</option>))}
                      </select>
                    </label>
                    <label className="label">Instant feed metabolome
                      <select className="input" value={d.feed_metabolome_instant_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_metabolome_instant_id: e.target.value || undefined } : x))}>
                        <option value="">None</option>
                        {metabolomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
                      </select>
                    </label>
                    <label className="label">Continuous feed metabolome
                      <select className="input" value={d.feed_metabolome_cont_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_metabolome_cont_id: e.target.value || undefined } : x))}>
                        <option value="">None</option>
                        {metabolomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
                      </select>
                    </label>
                  </div>
                  <div className="form-row" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }}>
                    <label className="label">Instant feed microbiome
                      <select className="input" value={d.feed_microbiome_instant_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_microbiome_instant_id: e.target.value || undefined } : x))}>
                        <option value="">None</option>
                        {microbiomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
                      </select>
                    </label>
                    <label className="label">Continuous feed microbiome
                      <select className="input" value={d.feed_microbiome_cont_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_microbiome_cont_id: e.target.value || undefined } : x))}>
                        <option value="">None</option>
                        {microbiomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
                      </select>
                    </label>
                  </div>
                  <div className="row" style={{ justifyContent: 'flex-end', gap: 8 }}>
                    <button className="btn" onClick={() => setPulseDrafts((prev) => prev.filter((x) => x.id !== d.id))}>Delete</button>
                    <button className="btn primary" disabled={!d.name.trim() || d.t_start >= d.t_end || !d.environment_id || savingPulseId === d.id} onClick={async () => {
                      const overlap = pulses.some((p) => !(d.t_end <= p.t_start || d.t_start >= p.t_end));
                      if (overlap) { alert('Pulse overlaps an existing saved pulse'); return; }
                      try {
                        setSavingPulseId(d.id);
                        const payload = { name: d.name.trim(), t_start: d.t_start, t_end: d.t_end, n_steps: d.n_steps, vin: d.vin, vout: d.vout, qin: d.qin, qout: d.qout, environment_id: d.environment_id!, feed_metabolome_instant_id: d.feed_metabolome_instant_id, feed_metabolome_cont_id: d.feed_metabolome_cont_id, feed_microbiome_instant_id: d.feed_microbiome_instant_id, feed_microbiome_cont_id: d.feed_microbiome_cont_id };
                        const saved = await createPulse(payload);
                        setPulses((prev) => [saved, ...prev].sort((a, b) => a.t_start - b.t_start));
                        setPulseDrafts((prev) => prev.filter((x) => x.id !== d.id));
                        setSelPulseId(null); setSelPulseKind(null);
                      } catch { alert('Failed to save pulse'); } finally { setSavingPulseId(null); }
                    }}>{savingPulseId === d.id ? 'Saving…' : 'Save'}</button>
                  </div>
                </>
              );
            })()
          ) : (
            (() => {
              const p = pulses.find((x) => x.id === selPulseId);
              if (!p) return <p className="muted">Pulse not found.</p>;
              return (
                <div className="form-row" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <label className="label">Name
                    <input className="input" value={p.name} onChange={(e) => setPulses((prev) => prev.map((x) => x.id === p.id ? { ...x, name: e.target.value } : x))} />
                  </label>
                  <button className="btn" onClick={async () => { try { await renamePulse(p.id, p.name); } catch { alert('Failed to rename pulse'); } }}>Save name</button>
                  <div style={{ flex: 1 }} />
                  <button className="btn" onClick={async () => { const ok = window.confirm(`Delete pulse "${p.name}"?`); if (!ok) return; try { await deletePulse(p.id); setPulses((prev) => prev.filter((x) => x.id !== p.id)); setSelPulseId(null); setSelPulseKind(null);} catch { alert('Failed to delete pulse'); } }}>Delete</button>
                </div>
              );
            })()
          )}
          </div>
        </div>

        {/* Collapsible saved pulses list */}
        <details style={{ marginTop: 12 }}>
          <summary className="muted" style={{ cursor: 'pointer' }}>Saved pulses ({pulses.length})</summary>
          <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button
                className="btn"
                disabled={pulses.length === 0}
                onClick={async () => {
                  if (pulses.length === 0) return;
                  const ok = window.confirm('Delete all saved pulses?');
                  if (!ok) return;
                  try {
                    const ids = pulses.map((p) => p.id);
                    await Promise.allSettled(ids.map((id) => deletePulse(id)));
                  } catch {}
                  setPulses([]);
                  setSelPulseId(null);
                  setSelPulseKind(null);
                }}
              >
                Delete all
              </button>
            </div>
            {pulses.length === 0 ? (
              <div className="muted">No saved pulses yet.</div>
            ) : (
              pulses.map((p) => (
                <div key={p.id} className="row" style={{ gap: 8, alignItems: 'center' }}>
                  <span className="chip" style={{ minWidth: 0 }}>{p.name || '(unnamed)'} · {p.t_start}–{p.t_end}h · {p.n_steps} steps</span>
                  <div style={{ flex: 1 }} />
                  <button className="btn" onClick={() => { setSelPulseId(p.id); setSelPulseKind('saved'); }}>Edit</button>
                </div>
              ))
            )}
          </div>
        </details>

        {/* Saved timelines (from backend) */}
        <details style={{ marginTop: 12 }}>
          <summary className="muted" style={{ cursor: 'pointer' }}>Saved timelines ({backendTimelines.length})</summary>
          <div style={{ marginTop: 8, display: 'grid', gap: 6 }}>
            {backendTimelines.length === 0 ? (
              <div className="muted">No saved timelines yet.</div>
            ) : (
              backendTimelines.map((t) => (
                <div key={t.id} className="row" style={{ gap: 8, alignItems: 'center' }}>
                  <span className="chip" style={{ minWidth: 0 }}>{t.name}</span>
                  <div className="muted">{t.t_start}–{t.t_end}h · {t.n_pulses} pulses</div>
                  <div style={{ flex: 1 }} />
                  <button className="btn" onClick={() => setSelectedTimelineId(t.id)}>Use</button>
                </div>
              ))
            )}
          </div>
        </details>
      </div>

      {cySpec && (
        <div className="card card--white" style={{ gridColumn: "span 12" }}>
          <h2>Model Graph</h2>
          {error && <div className="hint error">{error}</div>}
          <div style={{ height: 480 }}>
            <CytoscapeComponent
              elements={elementsArray}
              layout={cySpec.layout || { name: "preset" }}
              style={{ width: "100%", height: "100%" }}
              stylesheet={cySpec.style || cySpec.stylesheet || []}
            />
          </div>
        </div>
      )}

      {/* Pulses (list editor for now; timeline view to come) */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <div className="row space-between center">
          <h2>Pulses</h2>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn" onClick={() => setShowPulseList((v) => !v)}>{showPulseList ? 'Hide' : 'Inspect'}</button>
            <button
              className="btn"
              onClick={() => {
                // add 1h pulse at end
                const lastEnd = pulses.length > 0 ? Math.max(...pulses.map((p) => p.t_end)) : 0;
                const id = `pulse-draft-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`;
                setPulseDrafts((prev) => [
                  { id, name: '', t_start: lastEnd, t_end: lastEnd + 1, n_steps: 60, vin: 0, vout: 0, qin: 0, qout: 0 },
                  ...prev,
                ]);
              }}
            >New pulse</button>
          </div>
        </div>

        {showPulseList && (
          <>
          {pulseDrafts.length > 0 && (
          <div className="row" style={{ gap: 8, alignItems: 'center', marginTop: 8, fontSize: 12 }}>
            <span style={{ width: 8 }} />
            <span className="muted" style={{ width: 200 }}>Name</span>
            <span className="muted" style={{ width: 90 }}>Start (h)</span>
            <span className="muted" style={{ width: 90 }}>End (h)</span>
            <span className="muted" style={{ width: 90 }}>Steps</span>
            <span className="muted" style={{ width: 90 }}>vin</span>
            <span className="muted" style={{ width: 90 }}>vout</span>
            <span className="muted" style={{ width: 90 }}>qin</span>
            <span className="muted" style={{ width: 90 }}>qout</span>
            <span className="muted" style={{ width: 200 }}>Environment</span>
            <span className="muted" style={{ width: 200 }}>Instant feed metabolome</span>
            <span className="muted" style={{ width: 200 }}>Continuous feed metabolome</span>
            <span style={{ flex: 1 }} />
          </div>
          )}

        {pulseDrafts.map((d) => (
          <div key={d.id} className="row" style={{ gap: 8, alignItems: 'center', marginTop: 6 }}>
            <span aria-label="unsaved" title="Unsaved draft" style={{ width: 8, height: 8, borderRadius: 999, background: '#f87171', display: 'inline-block' }} />
            <input className="input" placeholder="Pulse name" value={d.name} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, name: e.target.value } : x))} style={{ width: 200 }} />
            <input className="input" type="number" step={0.25} min={0} value={d.t_start} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, t_start: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <input className="input" type="number" step={0.25} min={0} value={d.t_end} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, t_end: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <input className="input" type="number" step={1} min={1} value={d.n_steps} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, n_steps: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <input className="input" type="number" step={0.1} min={0} value={d.vin} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, vin: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <input className="input" type="number" step={0.1} min={0} value={d.vout} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, vout: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <input className="input" type="number" step={0.01} min={0} value={d.qin} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, qin: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <input className="input" type="number" step={0.01} min={0} value={d.qout} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, qout: Number(e.target.value) } : x))} style={{ width: 90 }} />
            <select className="input" value={d.environment_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, environment_id: e.target.value || undefined } : x))} style={{ width: 200 }}>
              <option value="">Select environment</option>
              {envs.map((e) => (<option key={e.id} value={e.id}>{e.name} · {e.metabolome_name}</option>))}
            </select>
            <select className="input" value={d.feed_metabolome_instant_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_metabolome_instant_id: e.target.value || undefined } : x))} style={{ width: 200 }}>
              <option value="">Instant feed metabolome</option>
              {metabolomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
            </select>
            <select className="input" value={d.feed_metabolome_cont_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_metabolome_cont_id: e.target.value || undefined } : x))} style={{ width: 200 }}>
              <option value="">Continuous feed metabolome</option>
              {metabolomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
            </select>
            <select className="input" value={d.feed_microbiome_instant_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_microbiome_instant_id: e.target.value || undefined } : x))} style={{ width: 200 }}>
              <option value="">Instant feed microbiome</option>
              {microbiomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
            </select>
            <select className="input" value={d.feed_microbiome_cont_id || ''} onChange={(e) => setPulseDrafts((prev) => prev.map((x) => x.id === d.id ? { ...x, feed_microbiome_cont_id: e.target.value || undefined } : x))} style={{ width: 200 }}>
              <option value="">Continuous feed microbiome</option>
              {microbiomes.map((m) => (<option key={m.id} value={m.id}>{m.name}</option>))}
            </select>
            <button className="btn primary" disabled={!d.name.trim() || d.t_start >= d.t_end || !d.environment_id || savingPulseId === d.id} onClick={async () => {
              // Enforce non-overlap vs saved pulses
              const overlap = pulses.some((p) => !(d.t_end <= p.t_start || d.t_start >= p.t_end));
              if (overlap) { alert('Pulse overlaps an existing saved pulse'); return; }
              try {
                setSavingPulseId(d.id);
                const payload = {
                  name: d.name.trim(), t_start: d.t_start, t_end: d.t_end, n_steps: d.n_steps,
                  vin: d.vin, vout: d.vout, qin: d.qin, qout: d.qout,
                  environment_id: d.environment_id!,
                  feed_metabolome_instant_id: d.feed_metabolome_instant_id,
                  feed_metabolome_cont_id: d.feed_metabolome_cont_id,
                };
                const saved = await createPulse(payload);
                setPulses((prev) => [saved, ...prev].sort((a, b) => a.t_start - b.t_start));
                setPulseDrafts((prev) => prev.filter((x) => x.id !== d.id));
              } catch (e) { alert('Failed to save pulse'); } finally { setSavingPulseId(null); }
            }}>{savingPulseId === d.id ? 'Saving…' : 'Save'}</button>
            <button className="btn" onClick={() => setPulseDrafts((prev) => prev.filter((x) => x.id !== d.id))}>Delete</button>
            <button className="btn" onClick={() => {
              const dur = Math.max(0.25, d.t_end - d.t_start);
              const start = d.t_end; // chain after
              const end = start + dur;
              const id = `pulse-draft-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`;
              const name = nextPulseName();
              const steps = Math.max(1, Math.round(dur * 60));
              const dup = { id, name, t_start: start, t_end: end, n_steps: steps, vin: d.vin, vout: d.vout, qin: d.qin, qout: d.qout, environment_id: d.environment_id, feed_metabolome_instant_id: d.feed_metabolome_instant_id, feed_metabolome_cont_id: d.feed_metabolome_cont_id, feed_microbiome_instant_id: d.feed_microbiome_instant_id, feed_microbiome_cont_id: d.feed_microbiome_cont_id };
              setPulseDrafts((prev) => [dup, ...prev]);
            }}>Duplicate</button>
          </div>
        ))}

        {pulses.map((p) => (
          <div key={p.id} className="row" style={{ gap: 8, alignItems: 'center', marginTop: 6 }}>
            <span aria-label="saved" title="Saved" style={{ width: 8, height: 8, borderRadius: 999, background: '#22d3ee', display: 'inline-block' }} />
            <input className="input" value={p.name} onChange={(e) => setPulses((prev) => prev.map((x) => x.id === p.id ? { ...x, name: e.target.value } as PulseOut : x))} style={{ width: 200 }} />
            <span className="muted" style={{ width: 90 }}>{p.t_start}</span>
            <span className="muted" style={{ width: 90 }}>{p.t_end}</span>
            <span className="muted" style={{ width: 90 }}>{p.n_steps}</span>
            <span className="muted" style={{ width: 90 }}>{p.vin}</span>
            <span className="muted" style={{ width: 90 }}>{p.vout}</span>
            <span className="muted" style={{ width: 90 }}>{p.qin}</span>
            <span className="muted" style={{ width: 90 }}>{p.qout}</span>
            <span className="chip" style={{ minWidth: 0 }}>{p.environment_name}</span>
            <span className="muted" style={{ minWidth: 0 }}>{p.feed_metabolome_instant_name || '-'}</span>
            <span className="muted" style={{ minWidth: 0 }}>{p.feed_metabolome_cont_name || '-'}</span>
            <div style={{ flex: 1 }} />
            <button className="btn" disabled={savingPulseId === p.id} onClick={async () => {
              try { setSavingPulseId(p.id); const up = await renamePulse(p.id, p.name); setPulses((prev) => prev.map((x) => x.id === p.id ? { ...x, name: up.name } : x)); } catch { alert('Failed to rename pulse'); } finally { setSavingPulseId(null); }
            }}>Save name</button>
            <button className="btn" disabled={savingPulseId === p.id} onClick={async () => {
              const ok = window.confirm(`Delete pulse "${p.name}"?`); if (!ok) return; try { setSavingPulseId(p.id); await deletePulse(p.id); setPulses((prev) => prev.filter((x) => x.id !== p.id)); } catch { alert('Failed to delete pulse'); } finally { setSavingPulseId(null); }
            }}>Delete</button>
            <button className="btn" onClick={() => {
              const dur = Math.max(0.25, p.t_end - p.t_start);
              const lastSaved = pulses.length > 0 ? Math.max(...pulses.map((q) => q.t_end)) : 0;
              const lastDraft = pulseDrafts.length > 0 ? Math.max(...pulseDrafts.map((q) => q.t_end)) : 0;
              const start = Math.max(lastSaved, lastDraft);
              const end = start + dur;
              const id = `pulse-draft-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,8)}`;
              const name = nextPulseName();
              const steps = Math.max(1, Math.round(dur * 60));
              const dup = { id, name, t_start: start, t_end: end, n_steps: steps, vin: p.vin, vout: p.vout, qin: p.qin, qout: p.qout, environment_id: (p as any).environment_id, feed_metabolome_instant_id: (p as any).feed_metabolome_instant_id, feed_metabolome_cont_id: (p as any).feed_metabolome_cont_id };
              setPulseDrafts((prev) => [dup, ...prev]);
            }}>Duplicate</button>
          </div>
        ))}
        </>
        )}
      </div>
    </div>
  );
}
