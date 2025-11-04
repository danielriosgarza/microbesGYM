const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export async function getCytoscape(signal?: AbortSignal, opts?: { microbiome_id?: string }): Promise<any> {
  const url = new URL(`${API_BASE}/api/viz/cytoscape`);
  if (opts?.microbiome_id) url.searchParams.set("microbiome_id", opts.microbiome_id);
  const res = await fetch(url.toString(), { signal });
  if (!res.ok) throw new Error(`getCytoscape failed: ${res.status}`);
  return await res.json();
}

