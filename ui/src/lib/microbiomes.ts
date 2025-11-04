const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface MicrobiomeCreate {
  name: string;
  metabolome_id: string;
  subpop_counts: Record<string, Record<string, number>>;
}

export interface MicrobiomeOut {
  id: string;
  name: string;
  n_species: number;
  n_subpops: number;
}

export async function listMicrobiomes(signal?: AbortSignal): Promise<MicrobiomeOut[]> {
  const res = await fetch(`${API_BASE}/api/microbiomes/`, { signal });
  if (!res.ok) throw new Error(`listMicrobiomes failed: ${res.status}`);
  return (await res.json()) as MicrobiomeOut[];
}

export async function createMicrobiome(payload: MicrobiomeCreate, signal?: AbortSignal): Promise<MicrobiomeOut> {
  const res = await fetch(`${API_BASE}/api/microbiomes/`, {
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
    throw new Error(`createMicrobiome failed: ${res.status}${detail ? ` ${detail}` : ""}`);
  }
  return (await res.json()) as MicrobiomeOut;
}

export async function renameMicrobiome(id: string, name: string, signal?: AbortSignal): Promise<MicrobiomeOut> {
  const res = await fetch(`${API_BASE}/api/microbiomes/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal,
  });
  if (!res.ok) throw new Error(`renameMicrobiome failed: ${res.status}`);
  return (await res.json()) as MicrobiomeOut;
}

export async function deleteMicrobiome(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/microbiomes/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deleteMicrobiome failed: ${res.status}`);
}

export async function getMicrobiomePlot(id: string, signal?: AbortSignal): Promise<{ data: any[]; layout: any; config?: any }> {
  const res = await fetch(`${API_BASE}/api/microbiomes/${encodeURIComponent(id)}/plot`, { signal });
  if (!res.ok) throw new Error(`getMicrobiomePlot failed: ${res.status}`);
  return (await res.json()) as { data: any[]; layout: any; config?: any };
}
