const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface FeedingTermIn {
  id?: string;
  metDict: Record<string, [number, number]>;
}

export interface SubpopulationIn {
  name: string;
  species: string;
  count: number;
  mumax: number;
  feedingTerms: FeedingTermIn[];
  pHopt: number;
  pH_sensitivity_left: number;
  pH_sensitivity_right: number;
  Topt: number;
  tempSensitivity_left: number;
  tempSensitivity_right: number;
  state: string;
  color: string;
}

export interface TransitionIn {
  target: string;
  condition: string;
  rate: number;
}

export interface BacteriaIn {
  species: string;
  color: string;
  subpopulations: SubpopulationIn[];
  connections: Record<string, TransitionIn[]>;
}

export interface BacteriaOut {
  id: string;
  species: string;
  n_subpops: number;
  subpop_names?: string[];
}

export async function listBacteria(signal?: AbortSignal): Promise<BacteriaOut[]> {
  const res = await fetch(`${API_BASE}/api/bacteria/`, { signal });
  if (!res.ok) throw new Error(`listBacteria failed: ${res.status}`);
  return (await res.json()) as BacteriaOut[];
}

export async function createBacteria(payload: BacteriaIn, signal?: AbortSignal): Promise<BacteriaOut> {
  const res = await fetch(`${API_BASE}/api/bacteria/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    let detail = "";
    try {
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const data = await res.json();
        const d = (data && (data.detail ?? data.message)) as unknown;
        if (typeof d === "string") detail = d;
        else if (d) detail = JSON.stringify(d);
      } else {
        detail = await res.text();
      }
    } catch {}
    throw new Error(`createBacteria failed: ${res.status}${detail ? ` ${detail}` : ""}`);
  }
  return (await res.json()) as BacteriaOut;
}

export async function renameBacteria(id: string, species: string, signal?: AbortSignal): Promise<BacteriaOut> {
  const res = await fetch(`${API_BASE}/api/bacteria/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ species }),
    signal,
  });
  if (!res.ok) throw new Error(`renameBacteria failed: ${res.status}`);
  return (await res.json()) as BacteriaOut;
}

export async function deleteBacteria(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/bacteria/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deleteBacteria failed: ${res.status}`);
}

