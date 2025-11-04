const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export type Formula = Record<string, number>;

export interface MetaboliteIn {
  name: string;
  concentration: number;
  formula: Formula;
  color: string;
  description?: string;
}

export interface MetaboliteOut extends MetaboliteIn {
  id: string;
  description: string;
}

export async function listMetabolites(signal?: AbortSignal): Promise<MetaboliteOut[]> {
  const res = await fetch(`${API_BASE}/api/metabolites/`, { signal });
  if (!res.ok) throw new Error(`listMetabolites failed: ${res.status}`);
  return (await res.json()) as MetaboliteOut[];
}

export async function createMetabolite(
  payload: MetaboliteIn,
  signal?: AbortSignal
): Promise<MetaboliteOut> {
  // Wrap fetch with a timeout so UI can't hang indefinitely if the request stalls
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 10000);
  // If caller provided a signal, link its abort to ours
  if (signal) {
    if (signal.aborted) {
      ctl.abort();
    } else {
      const onAbort = () => ctl.abort();
      signal.addEventListener("abort", onAbort, { once: true });
    }
  }
  try {
    const res = await fetch(`${API_BASE}/api/metabolites/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctl.signal,
    });
    if (!res.ok) throw new Error(`createMetabolite failed: ${res.status}`);
    return (await res.json()) as MetaboliteOut;
  } finally {
    clearTimeout(timer);
  }
}

export async function deleteMetabolite(id: string, signal?: AbortSignal): Promise<boolean> {
  const res = await fetch(`${API_BASE}/api/metabolites/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal,
  });
  if (res.status === 204) return true;
  if (res.status === 404) return false;
  throw new Error(`deleteMetabolite failed: ${res.status}`);
}

