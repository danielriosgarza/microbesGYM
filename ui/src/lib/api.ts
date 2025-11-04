const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export async function getHealth(signal?: AbortSignal): Promise<{ status: string } | null> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal });
    if (!res.ok) return null;
    return (await res.json()) as { status: string };
  } catch {
    return null;
  }
}

export interface PresetListItem {
  id: string;
  name: string;
  description?: string;
  tags?: string[];
}

export async function listPresets(signal?: AbortSignal): Promise<PresetListItem[]> {
  try {
    const res = await fetch(`${API_BASE}/api/presets/`, { signal });
    if (!res.ok) return [];
    return (await res.json()) as PresetListItem[];
  } catch {
    return [];
  }
}

export async function getPreset(id: string, signal?: AbortSignal): Promise<any | null> {
  try {
    const res = await fetch(`${API_BASE}/api/presets/${encodeURIComponent(id)}`, { signal });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

