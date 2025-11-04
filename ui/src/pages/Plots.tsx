import React from "react";
import { ApiStatus } from "../components/ApiStatus";
import { listMetabolomes, getMetabolomePlot, type MetabolomeOut } from "../lib/metabolomes";
import { listMicrobiomes, getMicrobiomePlot, type MicrobiomeOut } from "../lib/microbiomes";
import { listSimulations, getSimulation, type SimulationListItem, type SimulationResultOut } from "../lib/simulations";
import { ensurePlotly } from "../lib/plotly";

export function Plots() {
  const [metabolomes, setMetabolomes] = React.useState<MetabolomeOut[]>([]);
  const [selectedId, setSelectedId] = React.useState<string>("");
  const [plotSpec, setPlotSpec] = React.useState<{ data: any[]; layout: any; config?: any } | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const plotRef = React.useRef<HTMLDivElement | null>(null);

  const [microbiomes, setMicrobiomes] = React.useState<MicrobiomeOut[]>([]);
  const [selectedMicroId, setSelectedMicroId] = React.useState<string>("");
  const [microPlotSpec, setMicroPlotSpec] = React.useState<{ data: any[]; layout: any; config?: any } | null>(null);
  const microPlotRef = React.useRef<HTMLDivElement | null>(null);

  // Simulations
  const [simulations, setSimulations] = React.useState<SimulationListItem[]>([]);
  const [selectedSimId, setSelectedSimId] = React.useState<string>("");
  const [simPlotSpec, setSimPlotSpec] = React.useState<{ data: any[]; layout: any; config?: any } | null>(null);
  const simPlotRef = React.useRef<HTMLDivElement | null>(null);

  // Load list of metabolomes on mount
  const refreshList = React.useCallback(async () => {
    try {
      const list = await listMetabolomes();
      setMetabolomes(list);
      if (selectedId && !list.find((m) => m.id === selectedId)) {
        setSelectedId("");
        setPlotSpec(null);
      }
    } catch {
      setMetabolomes([]);
    }
  }, [selectedId]);

  React.useEffect(() => {
    void refreshList();
  }, [refreshList]);

  // React to metabolomes created/renamed/deleted elsewhere
  React.useEffect(() => {
    const handler = () => { void refreshList(); };
    window.addEventListener("metabolomes:changed", handler);
    return () => window.removeEventListener("metabolomes:changed", handler);
  }, [refreshList]);

  // Load microbiomes list on mount
  const refreshMicrobiomes = React.useCallback(async () => {
    try {
      const list = await listMicrobiomes();
      setMicrobiomes(list);
      if (selectedMicroId && !list.find((m) => m.id === selectedMicroId)) {
        setSelectedMicroId("");
        setMicroPlotSpec(null);
      }
    } catch {
      setMicrobiomes([]);
    }
  }, [selectedMicroId]);

  React.useEffect(() => { void refreshMicrobiomes(); }, [refreshMicrobiomes]);
  React.useEffect(() => {
    const handler = () => { void refreshMicrobiomes(); };
    window.addEventListener('microbiomes:changed', handler);
    return () => window.removeEventListener('microbiomes:changed', handler);
  }, [refreshMicrobiomes]);

  // Load simulations list on mount and when a simulation completes
  const refreshSimulations = React.useCallback(async (preferId?: string) => {
    try {
      const list = await listSimulations();
      setSimulations(list);
      if (preferId && list.find((s) => s.id === preferId)) {
        setSelectedSimId(preferId);
        try {
          const res: SimulationResultOut = await getSimulation(preferId);
          setSimPlotSpec(res.plot);
        } catch {}
      } else if (selectedSimId && !list.find((s) => s.id === selectedSimId)) {
        setSelectedSimId("");
        setSimPlotSpec(null);
      }
    } catch {
      setSimulations([]);
    }
  }, [selectedSimId]);

  React.useEffect(() => { void refreshSimulations(); }, [refreshSimulations]);
  React.useEffect(() => {
    const onCompleted = (e: Event) => {
      try {
        const detail = (e as CustomEvent).detail as { id?: string } | undefined;
        void refreshSimulations(detail?.id);
      } catch {
        void refreshSimulations();
      }
    };
    window.addEventListener('simulation:completed', onCompleted as EventListener);
    return () => window.removeEventListener('simulation:completed', onCompleted as EventListener);
  }, [refreshSimulations]);

  // Render plot when spec or selection changes
  React.useEffect(() => {
    (async () => {
      if (!plotSpec || !plotRef.current) return;
      try {
        const Plotly = await ensurePlotly();
        await Plotly.react(plotRef.current, plotSpec.data, plotSpec.layout, plotSpec.config || { responsive: true });
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    })();
  }, [plotSpec]);

  React.useEffect(() => {
    (async () => {
      if (!microPlotSpec || !microPlotRef.current) return;
      try {
        const Plotly = await ensurePlotly();
        await Plotly.react(microPlotRef.current, microPlotSpec.data, microPlotSpec.layout, microPlotSpec.config || { responsive: true });
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    })();
  }, [microPlotSpec]);

  React.useEffect(() => {
    (async () => {
      if (!simPlotSpec || !simPlotRef.current) return;
      try {
        const Plotly = await ensurePlotly();
        await Plotly.react(simPlotRef.current, simPlotSpec.data, simPlotSpec.layout, simPlotSpec.config || { responsive: true });
      } catch (e) {
        // eslint-disable-next-line no-console
        console.error(e);
      }
    })();
  }, [simPlotSpec]);

  const handleSelect = async (id: string) => {
    setSelectedId(id);
    if (!id) { setPlotSpec(null); return; }
    setLoading(true);
    setError(null);
    try {
      const spec = await getMetabolomePlot(id);
      setPlotSpec(spec);
    } catch (e) {
      setError("Failed to load plot");
      setPlotSpec(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSimulation = async (id: string) => {
    setSelectedSimId(id);
    if (!id) { setSimPlotSpec(null); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await getSimulation(id);
      setSimPlotSpec(res.plot);
    } catch (e) {
      setError("Failed to load simulation plot");
      setSimPlotSpec(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectMicro = async (id: string) => {
    setSelectedMicroId(id);
    if (!id) { setMicroPlotSpec(null); return; }
    setLoading(true);
    setError(null);
    try {
      const spec = await getMicrobiomePlot(id);
      setMicroPlotSpec(spec);
    } catch (e) {
      setError("Failed to load microbiome plot");
      setMicroPlotSpec(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid" id="panel-plots">
      <div className="card" style={{ gridColumn: "span 12" }}>
        <ApiStatus />
      </div>

      {/* Microbiomes first */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Microbiomes</h2>
        {microbiomes.length === 0 ? (
          <p className="muted">No microbiomes yet. Create one from the Simulation tab.</p>
        ) : (
          <div className="row" style={{ alignItems: 'center', gap: 8 }}>
            <label className="label" htmlFor="sel-micro">Select microbiome</label>
            <select id="sel-micro" className="input" value={selectedMicroId} onChange={(e) => handleSelectMicro(e.target.value)}>
              <option value="">-- Choose --</option>
              {microbiomes.map((m) => (
                <option key={m.id} value={m.id}>{m.name} ({m.n_species} sp, {m.n_subpops} subpops)</option>
              ))}
            </select>
            <button className="btn" onClick={async () => {
              try { setMicrobiomes(await listMicrobiomes()); } catch {}
            }}>Refresh</button>
          </div>
        )}
      </div>

      {selectedMicroId && microPlotSpec && (
        <div className="card card--white" style={{ gridColumn: "span 12" }}>
          <div className="row space-between center">
            <h2>Microbiome Plot</h2>
            {loading ? <span className="muted">Loading...</span> : null}
          </div>
          {error && <div className="hint error">{error}</div>}
          <div ref={microPlotRef} style={{ height: 600 }} />
        </div>
      )}

      {/* Metabolomes second */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Metabolomes</h2>
        {metabolomes.length === 0 ? (
          <p className="muted">No metabolomes yet. Create one from the Simulation tab.</p>
        ) : (
          <div className="row" style={{ alignItems: 'center', gap: 8 }}>
            <label className="label" htmlFor="sel">Select metabolome</label>
            <select id="sel" className="input" value={selectedId} onChange={(e) => handleSelect(e.target.value)}>
              <option value="">-- Choose --</option>
              {metabolomes.map((m) => (
                <option key={m.id} value={m.id}>{m.name} ({m.n_metabolites})</option>
              ))}
            </select>
            <button className="btn" onClick={async () => {
              try {
                const list = await listMetabolomes();
                setMetabolomes(list);
              } catch {}
            }}>Refresh</button>
          </div>
        )}
      </div>

      {selectedId && plotSpec && (
        <div className="card card--white" style={{ gridColumn: "span 12" }}>
          <div className="row space-between center">
            <h2>Concentration Plot</h2>
            {loading ? <span className="muted">Loading...</span> : null}
          </div>
          {error && <div className="hint error">{error}</div>}
          <div ref={plotRef} style={{ height: 520 }} />
        </div>
      )}

      {/* Simulations last */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h2>Simulations</h2>
        {simulations.length === 0 ? (
          <p className="muted">No simulations yet. Run one from the Simulation tab.</p>
        ) : (
          <div className="row" style={{ alignItems: 'center', gap: 8 }}>
            <label className="label" htmlFor="sel-sim">Select simulation</label>
            <select id="sel-sim" className="input" value={selectedSimId} onChange={(e) => void handleSelectSimulation(e.target.value)}>
              <option value="">-- Choose --</option>
              {simulations.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
            <button className="btn" onClick={() => void refreshSimulations(selectedSimId)}>Refresh</button>
          </div>
        )}
      </div>

      {selectedSimId && simPlotSpec && (
        <div className="card card--white" style={{ gridColumn: "span 12" }}>
          <div className="row space-between center">
            <h2>Simulation Results</h2>
            {loading ? <span className="muted">Loading...</span> : null}
          </div>
          {error && <div className="hint error">{error}</div>}
          <div ref={simPlotRef} style={{ height: 640 }} />
        </div>
      )}
    </div>
  );
}
