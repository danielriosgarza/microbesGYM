const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface PulseCreateIn {
  name: string;
  t_start: number;
  t_end: number;
  n_steps: number;
  vin: number;
  vout: number;
  qin: number;
  qout: number;
  environment_id: string;
  feed_metabolome_instant_id?: string;
  feed_metabolome_cont_id?: string;
  // New: optional microbiome feeds
  feed_microbiome_instant_id?: string;
  feed_microbiome_cont_id?: string;
}

export interface PulseOut {
  id: string;
  name: string;
  t_start: number;
  t_end: number;
  n_steps: number;
  vin: number;
  vout: number;
  qin: number;
  qout: number;
  environment_id: string;
  environment_name: string;
  feed_metabolome_instant_id?: string;
  feed_metabolome_instant_name?: string;
  feed_metabolome_cont_id?: string;
  feed_metabolome_cont_name?: string;
  // New: microbiome feeds
  feed_microbiome_instant_id?: string;
  feed_microbiome_instant_name?: string;
  feed_microbiome_cont_id?: string;
  feed_microbiome_cont_name?: string;
}

export type PulseDetail = PulseOut;

export async function listPulses(signal?: AbortSignal): Promise<PulseOut[]> {
  const res = await fetch(`${API_BASE}/api/pulses/`, { signal });
  if (!res.ok) throw new Error(`listPulses failed: ${res.status}`);
  return (await res.json()) as PulseOut[];
}

export async function createPulse(payload: PulseCreateIn, signal?: AbortSignal): Promise<PulseOut> {
  const res = await fetch(`${API_BASE}/api/pulses/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`createPulse failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as PulseOut;
}

export async function renamePulse(id: string, name: string, signal?: AbortSignal): Promise<PulseOut> {
  const res = await fetch(`${API_BASE}/api/pulses/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`renamePulse failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as PulseOut;
}

export async function deletePulse(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/pulses/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deletePulse failed: ${res.status}`);
}

