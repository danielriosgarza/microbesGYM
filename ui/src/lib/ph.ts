const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface PHFunctionCreateIn {
  name: string;
  metabolome_id: string;
  baseValue: number;
  weights: Record<string, number>;
}

export interface PHFunctionOut {
  id: string;
  name: string;
  metabolome_id: string;
  metabolome_name: string;
  n_metabolites: number;
}

export interface PHFunctionDetail extends PHFunctionOut {
  baseValue: number;
  weights: Record<string, number>;
}

export async function listPHFunctions(signal?: AbortSignal): Promise<PHFunctionOut[]> {
  const res = await fetch(`${API_BASE}/api/ph_functions/`, { signal });
  if (!res.ok) throw new Error(`listPHFunctions failed: ${res.status}`);
  return (await res.json()) as PHFunctionOut[];
}

export async function getPHFunction(id: string, signal?: AbortSignal): Promise<PHFunctionDetail> {
  const res = await fetch(`${API_BASE}/api/ph_functions/${encodeURIComponent(id)}`, { signal });
  if (!res.ok) throw new Error(`getPHFunction failed: ${res.status}`);
  return (await res.json()) as PHFunctionDetail;
}

export async function createPHFunction(payload: PHFunctionCreateIn, signal?: AbortSignal): Promise<PHFunctionOut> {
  const res = await fetch(`${API_BASE}/api/ph_functions/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`createPHFunction failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as PHFunctionOut;
}

export async function renamePHFunction(id: string, name: string, signal?: AbortSignal): Promise<PHFunctionOut> {
  const res = await fetch(`${API_BASE}/api/ph_functions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`renamePHFunction failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as PHFunctionOut;
}

export async function deletePHFunction(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/ph_functions/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deletePHFunction failed: ${res.status}`);
}

