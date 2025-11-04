const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface SimulationRunIn {
  name: string;
  metabolome_id: string;
  microbiome_id: string;
  timeline_id: string;
  volume: number;
  mode: "fast" | "balanced" | "accurate";
}

export interface SimulationListItem {
  id: string;
  name: string;
}

export interface SimulationResultOut {
  id: string;
  name: string;
  summary: Record<string, number>;
  plot: { data: any[]; layout: any; config?: any };
}

export async function runSimulation(payload: SimulationRunIn, signal?: AbortSignal): Promise<SimulationResultOut> {
  const res = await fetch(`${API_BASE}/api/simulations/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`runSimulation failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as SimulationResultOut;
}

export async function listSimulations(signal?: AbortSignal): Promise<SimulationListItem[]> {
  const res = await fetch(`${API_BASE}/api/simulations/`, { signal });
  if (!res.ok) throw new Error(`listSimulations failed: ${res.status}`);
  return (await res.json()) as SimulationListItem[];
}

export async function getSimulation(simulation_id: string, signal?: AbortSignal): Promise<SimulationResultOut> {
  const res = await fetch(`${API_BASE}/api/simulations/${encodeURIComponent(simulation_id)}`, { signal });
  if (!res.ok) throw new Error(`getSimulation failed: ${res.status}`);
  return (await res.json()) as SimulationResultOut;
}

export async function deleteSimulation(simulation_id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/simulations/${encodeURIComponent(simulation_id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deleteSimulation failed: ${res.status}`);
}


