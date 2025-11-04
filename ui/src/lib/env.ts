const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface EnvironmentCreateIn {
  name: string;
  ph_function_id: string;
  temperature: number;
  stirring_rate: number;
  stirring_base_std: number;
}

export interface EnvironmentOut {
  id: string;
  name: string;
  ph_function_id: string;
  ph_function_name: string;
  metabolome_id: string;
  metabolome_name: string;
  temperature: number;
  stirring_rate: number;
  stirring_base_std: number;
}

export type EnvironmentDetail = EnvironmentOut;

export async function listEnvironments(signal?: AbortSignal): Promise<EnvironmentOut[]> {
  const res = await fetch(`${API_BASE}/api/environments/`, { signal });
  if (!res.ok) throw new Error(`listEnvironments failed: ${res.status}`);
  return (await res.json()) as EnvironmentOut[];
}

export async function createEnvironment(payload: EnvironmentCreateIn, signal?: AbortSignal): Promise<EnvironmentOut> {
  const res = await fetch(`${API_BASE}/api/environments/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`createEnvironment failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as EnvironmentOut;
}

export async function renameEnvironment(id: string, name: string, signal?: AbortSignal): Promise<EnvironmentOut> {
  const res = await fetch(`${API_BASE}/api/environments/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`renameEnvironment failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as EnvironmentOut;
}

export async function deleteEnvironment(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/environments/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deleteEnvironment failed: ${res.status}`);
}

