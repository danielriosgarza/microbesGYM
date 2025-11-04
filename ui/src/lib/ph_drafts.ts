const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface PHDraftCreateIn {
  name: string;
  baseValue: number;
  weights: Record<string, number>;
}

export interface PHDraftOut {
  id: string;
  name: string;
  baseValue: number;
  n_weights: number;
}

export interface PHDraftDetail {
  id: string;
  name: string;
  baseValue: number;
  weights: Record<string, number>;
}

export async function listPHDrafts(signal?: AbortSignal): Promise<PHDraftOut[]> {
  const res = await fetch(`${API_BASE}/api/ph_drafts/`, { signal });
  if (!res.ok) return [];
  return (await res.json()) as PHDraftOut[];
}

export async function createPHDraft(payload: PHDraftCreateIn, signal?: AbortSignal): Promise<PHDraftOut> {
  const res = await fetch(`${API_BASE}/api/ph_drafts/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`createPHDraft failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as PHDraftOut;
}

export async function deletePHDraft(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/ph_drafts/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deletePHDraft failed: ${res.status}`);
}

export async function getPHDraft(id: string, signal?: AbortSignal): Promise<PHDraftDetail> {
  const res = await fetch(`${API_BASE}/api/ph_drafts/${encodeURIComponent(id)}`, { signal });
  if (!res.ok) throw new Error(`getPHDraft failed: ${res.status}`);
  return (await res.json()) as PHDraftDetail;
}


