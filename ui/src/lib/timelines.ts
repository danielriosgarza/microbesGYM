const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export interface TimelineCreateIn {
  name: string;
  pulse_ids: string[];
}

export interface TimelineOut {
  id: string;
  name: string;
  n_pulses: number;
  t_start: number;
  t_end: number;
}

export interface TimelineDetail {
  id: string;
  name: string;
  pulses: Array<{
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
    feed_microbiome_instant_id?: string;
    feed_microbiome_instant_name?: string;
    feed_microbiome_cont_id?: string;
    feed_microbiome_cont_name?: string;
  }>;
}

export async function listTimelines(signal?: AbortSignal): Promise<TimelineOut[]> {
  const res = await fetch(`${API_BASE}/api/timelines/`, { signal });
  if (!res.ok) throw new Error(`listTimelines failed: ${res.status}`);
  return (await res.json()) as TimelineOut[];
}

export async function createTimeline(payload: TimelineCreateIn, signal?: AbortSignal): Promise<TimelineOut> {
  const res = await fetch(`${API_BASE}/api/timelines/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`createTimeline failed: ${res.status}${txt ? ` ${txt}` : ""}`);
  }
  return (await res.json()) as TimelineOut;
}

export async function getTimelineDetail(id: string, signal?: AbortSignal): Promise<TimelineDetail> {
  const res = await fetch(`${API_BASE}/api/timelines/${encodeURIComponent(id)}`, { signal });
  if (!res.ok) throw new Error(`getTimelineDetail failed: ${res.status}`);
  return (await res.json()) as TimelineDetail;
}

export async function renameTimeline(id: string, name: string, signal?: AbortSignal): Promise<TimelineOut> {
  const res = await fetch(`${API_BASE}/api/timelines/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
    signal,
  });
  if (!res.ok) throw new Error(`renameTimeline failed: ${res.status}`);
  return (await res.json()) as TimelineOut;
}

export async function deleteTimeline(id: string, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_BASE}/api/timelines/${encodeURIComponent(id)}`, { method: "DELETE", signal });
  if (!res.ok) throw new Error(`deleteTimeline failed: ${res.status}`);
}


