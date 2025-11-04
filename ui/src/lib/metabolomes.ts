const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface MetabolomeIn {
  name: string;
  concentrations: Record<string, number>;
}

export interface MetabolomeOut {
  id: string;
  name: string;
  n_metabolites: number;
}

export async function listMetabolomes(signal?: AbortSignal): Promise<MetabolomeOut[]> {
  const res = await fetch(`${API_BASE}/api/metabolomes/`, { signal });
  if (!res.ok) throw new Error(`listMetabolomes failed: ${res.status}`);
  return (await res.json()) as MetabolomeOut[];
}

export async function createMetabolome(payload: MetabolomeIn, signal?: AbortSignal): Promise<MetabolomeOut> {
  const res = await fetch(`${API_BASE}/api/metabolomes/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    try {
      const body = await res.json();
      const detail = (body && (body.detail || body.message)) || JSON.stringify(body);
      throw new Error(`createMetabolome failed: ${res.status} ${detail}`);
    } catch {
      const text = await res.text().catch(() => "");
      throw new Error(`createMetabolome failed: ${res.status}${text ? ` ${text}` : ""}`);
    }
  }
  return (await res.json()) as MetabolomeOut;
}

export async function getMetabolomePlot(id: string, signal?: AbortSignal): Promise<{ data: any[]; layout: any; config?: any }> {
  const res = await fetch(`${API_BASE}/api/metabolomes/${encodeURIComponent(id)}/plot`, { signal });
  if (!res.ok) throw new Error(`getMetabolomePlot failed: ${res.status}`);
  return (await res.json()) as { data: any[]; layout: any; config?: any };
}

export async function deleteMetabolome(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/metabolomes/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal,
  });
  if (!res.ok) throw new Error(`deleteMetabolome failed: ${res.status}`);
}

export async function updateMetabolomeName(id: string, name: string, signal?: AbortSignal): Promise<MetabolomeOut> {
  const res = await fetch(`${API_BASE}/api/metabolomes/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`updateMetabolomeName failed: ${res.status}${text ? ` ${text}` : ""}`);
  }
  return (await res.json()) as MetabolomeOut;
}
