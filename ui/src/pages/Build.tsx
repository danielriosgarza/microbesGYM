import React from "react";
import { ApiStatus } from "../components/ApiStatus";
import { BuildCanvas, type RFNode } from "../components/BuildCanvas";
import { ReactFlowProvider } from "reactflow";
import { PHInspector, FeedingTermInspector, SubpopulationInspector, TransitionInspector, BacteriaInspector, MetaboliteInspector } from "./build_tab/inspectors";
import {
  createMetabolite,
  listMetabolites,
  deleteMetabolite,
  type MetaboliteIn,
  type MetaboliteOut,
} from "../lib/metabolites";
import { listMetabolomes, deleteMetabolome } from "../lib/metabolomes";
import { listMicrobiomes, deleteMicrobiome } from "../lib/microbiomes";
import { listPHFunctions, deletePHFunction } from "../lib/ph";
import { createPHDraft, listPHDrafts, deletePHDraft } from "../lib/ph_drafts";
import { listEnvironments, deleteEnvironment } from "../lib/env";
import { listPulses, deletePulse } from "../lib/pulses";
import { listTimelines, deleteTimeline } from "../lib/timelines";
import { createBacteria, listBacteria, deleteBacteria, type BacteriaIn, type SubpopulationIn, type FeedingTermIn, type TransitionIn } from "../lib/bacteria";
import { listPresets, getPreset } from "../lib/api";
import { listSimulations, deleteSimulation } from "../lib/simulations";

type Node = {
  localId: string;
  serverId?: string;
  saved: boolean;
  x: number;
  y: number;
  data: MetaboliteIn;
  lastSaved?: MetaboliteOut;
};

const defaultMetabolite = (): MetaboliteIn => ({
  name: "metabolite",
  concentration: 1,
  formula: { C: 0, H: 0, O: 0, N: 0, S: 0, P: 0 },
  color: "#22d3ee",
  description: "",
});

// Predefined metabolites for quick insertion
const predefinedMetabolites = [
  { name: 'Glucose', color: '#ff0000', concentration: 1, unit: 'mM', formula: { C: 6, H: 12, O: 6, N: 0, S: 0, P: 0 } },
  { name: 'Lactate', color: '#00ffaf', concentration: 1, unit: 'mM', formula: { C: 3, H: 6, O: 3, N: 0, S: 0, P: 0 } },
  { name: 'Acetate', color: '#003eff', concentration: 1, unit: 'mM', formula: { C: 2, H: 4, O: 2, N: 0, S: 0, P: 0 } },
  { name: 'Butyrate', color: '#ff00a1', concentration: 1, unit: 'mM', formula: { C: 4, H: 8, O: 2, N: 0, S: 0, P: 0 } },
  { name: 'Propionate', color: '#f97316', concentration: 1, unit: 'mM', formula: { C: 3, H: 6, O: 2, N: 0, S: 0, P: 0 } },
  { name: 'Succinate', color: '#00ff26', concentration: 1, unit: 'mM', formula: { C: 4, H: 6, O: 4, N: 0, S: 0, P: 0 } },
  { name: 'Formate', color: '#00c6ff', concentration: 1, unit: 'mM', formula: { C: 1, H: 2, O: 2, N: 0, S: 0, P: 0 } },
  { name: 'Hydrogen', color: '#e2e8f0', concentration: 1, unit: 'mM', formula: { C: 0, H: 2, O: 0, N: 0, S: 0, P: 0 } },
  { name: 'Pyruvate', color: '#ff8900', concentration: 1, unit: 'mM', formula: { C: 3, H: 4, O: 3, N: 0, S: 0, P: 0 } },
  { name: 'Fumarate', color: '#8b5cf6', concentration: 1, unit: 'mM', formula: { C: 4, H: 4, O: 4, N: 0, S: 0, P: 0 } },
  { name: 'Malate', color: '#10b981', concentration: 1, unit: 'mM', formula: { C: 4, H: 6, O: 5, N: 0, S: 0, P: 0 } },
  { name: 'Citrate', color: '#f59e0b', concentration: 1, unit: 'mM', formula: { C: 6, H: 8, O: 7, N: 0, S: 0, P: 0 } },
  { name: 'Isocitrate', color: '#ef4444', concentration: 1, unit: 'mM', formula: { C: 6, H: 8, O: 7, N: 0, S: 0, P: 0 } },
  { name: 'alpha-Ketoglutarate', color: '#06b6d4', concentration: 1, unit: 'mM', formula: { C: 5, H: 6, O: 5, N: 0, S: 0, P: 0 } },
  { name: 'Oxaloacetate', color: '#84cc16', concentration: 1, unit: 'mM', formula: { C: 4, H: 4, O: 5, N: 0, S: 0, P: 0 } },
  { name: 'Glycerol', color: '#64748b', concentration: 1, unit: 'mM', formula: { C: 3, H: 8, O: 3, N: 0, S: 0, P: 0 } },
  { name: 'Ethanol', color: '#a78bfa', concentration: 1, unit: 'mM', formula: { C: 2, H: 6, O: 1, N: 0, S: 0, P: 0 } },
  { name: 'Methanol', color: '#fb7185', concentration: 1, unit: 'mM', formula: { C: 1, H: 4, O: 1, N: 0, S: 0, P: 0 } },
  { name: 'Ammonia', color: '#e2e8f0', concentration: 1, unit: 'mM', formula: { C: 0, H: 3, O: 0, N: 1, S: 0, P: 0 } },
  { name: 'Nitrate', color: '#06b6d4', concentration: 1, unit: 'mM', formula: { C: 0, H: 0, O: 3, N: 1, S: 0, P: 0 } },
  { name: 'Sulfate', color: '#84cc16', concentration: 1, unit: 'mM', formula: { C: 0, H: 0, O: 4, N: 0, S: 1, P: 0 } },
  { name: 'Phosphate', color: '#fbbf24', concentration: 1, unit: 'mM', formula: { C: 0, H: 0, O: 4, N: 0, S: 0, P: 1 } },
  { name: 'Carbon Dioxide', color: '#94a3b8', concentration: 1, unit: 'mM', formula: { C: 1, H: 0, O: 2, N: 0, S: 0, P: 0 } },
  { name: 'Water', color: '#60a5fa', concentration: 1, unit: 'mM', formula: { C: 0, H: 2, O: 1, N: 0, S: 0, P: 0 } },
  { name: 'Trehalose', color: '#8900ff', concentration: 1, unit: 'mM', formula: { C: 12, H: 22, O: 11, N: 0, S: 0, P: 0 } },
  { name: 'Glutamate', color: '#00B8FF', concentration: 1, unit: 'mM', formula: { C: 5, H: 9, O: 4, N: 1, S: 0, P: 0 } },
  { name: 'Alanine', color: '#f59e0b', concentration: 1, unit: 'mM', formula: { C: 3, H: 7, O: 2, N: 1, S: 0, P: 0 } },
  { name: 'Glycine', color: '#10b981', concentration: 1, unit: 'mM', formula: { C: 2, H: 5, O: 2, N: 1, S: 0, P: 0 } },
  { name: 'Serine', color: '#ef4444', concentration: 1, unit: 'mM', formula: { C: 3, H: 7, O: 3, N: 1, S: 0, P: 0 } },
  { name: 'Aspartate', color: '#06b6d4', concentration: 1, unit: 'mM', formula: { C: 4, H: 7, O: 4, N: 1, S: 0, P: 0 } },
  { name: 'Mannose', color: '#024059', concentration: 1, unit: 'mM', formula: { C: 6, H: 12, O: 6, N: 0, S: 0, P: 0 } },
];

export function Build() {
  const [nodes, setNodes] = React.useState<Node[]>([]);
  const [rfNodes, setRfNodes] = React.useState<RFNode[]>([]);
  const [savingMets, setSavingMets] = React.useState(false);
  const [centerNodeId, setCenterNodeId] = React.useState<string | null>(null);
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [savedAt, setSavedAt] = React.useState<number | null>(null);
  const [showAddMenu, setShowAddMenu] = React.useState(false);
  const [showMetaboliteSub, setShowMetaboliteSub] = React.useState(false);
  const [clearing, setClearing] = React.useState(false);
  const [nuking, setNuking] = React.useState(false);
  const [presets, setPresets] = React.useState<Array<{ id: string; name: string }>>([]);
  const [ph, setPh] = React.useState<{ id: string; baseValue: number; weights: Record<string, number>; saved?: boolean } | null>(null);
  // Canvas color (user-selected). Default to white.
  const [canvasBg, setCanvasBg] = React.useState<string>("#ffffff");
  // Compute a grid color with contrast based on background brightness (hex only)
  const gridFor = React.useCallback((bg: string): string => {
    // If it's a gradient or non-hex value, fall back to a neutral dark grid
    if (!bg || !bg.startsWith('#') || (bg.length !== 7 && bg.length !== 4)) {
      return "rgba(0,0,0,0.12)";
    }
    const hex = bg.length === 4
      ? `#${bg[1]}${bg[1]}${bg[2]}${bg[2]}${bg[3]}${bg[3]}`
      : bg;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000; // 0-255
    // Light background → dark grid; dark background → light grid
    return brightness > 160 ? "rgba(0,0,0,0.14)" : "rgba(255,255,255,0.16)";
  }, []);
  // Feeding terms (by RF node id)
  const [feedingTerms, setFeedingTerms] = React.useState<Record<string, {
    name: string;
    inputs: Record<string, { yield: number; monodK: number }>;
    outputs: Record<string, { yield: number }>;
  }>>({});
  
  // Track edges for proper feeding term connections
  const [edges, setEdges] = React.useState<Array<{ id: string; source: string; target: string }>>([]);
  const [initialEdges, setInitialEdges] = React.useState<Array<{ id: string; source: string; target: string }>>([]);

  const handleExportJson = () => {
    // ----- Metabolome -----
    const metabolites: any[] = [];
    nodes.forEach((n) => {
      const d = n.data;
      const item: any = {
        name: d.name,
        concentration: d.concentration,
        formula: d.formula,
        color: d.color,
      };
      if (d.description && d.description.trim()) item.description = d.description;
      metabolites.push(item);
    });
    const model: any = { version: "1.0.0", metabolome: { metabolites, temperature: 37, stirring: 1 } };
    if (ph) {
      const base = Math.max(0, Math.min(14, ph.baseValue));
      const connected: Record<string, number> = {};
      for (const [k, v] of Object.entries(ph.weights)) {
        const num = Number(v);
        if (Number.isFinite(num) && num !== 0) connected[k] = num;
      }
      model.metabolome.pH = {
        name: "pH Control",
        baseValue: base,
        ...(Object.keys(connected).length ? { connectedMetabolites: connected } : {}),
        color: "#10b981",
        description: "pH control node that can be influenced by metabolites",
      };
    }

    // Helpers
    const rfById = (id: string) => rfNodes.find((n) => n.id === id);
    const idIsType = (id: string, type: string) => rfNodes.find((n) => n.id === id)?.type === type;
    const metNameById = (id: string) => nodes.find((n) => n.localId === id)?.data.name || id;

    // Unique name helper per group
    const ensureUnique = (name: string, seen: Set<string>) => {
      let nm = String(name || "").trim() || "unnamed";
      if (!seen.has(nm)) { seen.add(nm); return nm; }
      let i = 1; let cand = `${nm}_${i}`;
      while (seen.has(cand)) { i += 1; cand = `${nm}_${i}`; }
      seen.add(cand); return cand;
    };

    // ----- Microbiome from canvas -----
    // Build bacteria keyed by species
    const bacNodes = rfNodes.filter((n) => n.type === 'bacteriaNode');
    if (bacNodes.length > 0) {
      const seenSpecies = new Set<string>();
      const microbiome: any = { name: 'microbial_community', color: '#4444ff', bacteria: {} as Record<string, any> };

      for (const b of bacNodes) {
        const d: any = b.data || {};
        const rawSpecies = String(d.species || b.id);
        const species = ensureUnique(rawSpecies, seenSpecies);
        const color = String(d.color || '#54f542');

        // Subpopulations connected to this bacteria
        const spIds = edges
          .filter((e) => e.target === b.id && idIsType(e.source, 'subpopulationNode'))
          .map((e) => e.source);

        // Build subpopulations dict keyed by (unique) name
        const subpops: Record<string, any> = {};
        const spNameById: Record<string, string> = {};
        const seenSp = new Set<string>();
        for (const sid of spIds) {
          const sn = rfById(sid);
          if (!sn) continue;
          const sd: any = sn.data || {};
          const finalName = ensureUnique(String(sd.name || sid), seenSp);
          spNameById[sid] = finalName;
          subpops[finalName] = {
            name: finalName,
            count: Number(sd.count || 0) || 0,
            species,
            mumax: Number(sd.mumax || 0) || 0,
            pHopt: Number(sd.pHopt || 7.0) || 7.0,
            pH_sensitivity_left: Number(sd.pH_sensitivity_left || 2.0) || 2.0,
            pH_sensitivity_right: Number(sd.pH_sensitivity_right || 2.0) || 2.0,
            Topt: Number(sd.Topt || 37.0) || 37.0,
            tempSensitivity_left: Number(sd.tempSensitivity_left || 5.0) || 5.0,
            tempSensitivity_right: Number(sd.tempSensitivity_right || 2.0) || 2.0,
            state: String(sd.state || 'active'),
            color: String(sd.color || '#aaaaaa'),
            feedingTerms: [] as any[],
          };
        }

        // Attach feeding terms to subpops: ft -> sp edges
        for (const sid of spIds) {
          const spName = spNameById[sid];
          if (!spName) continue;
          const ftIds = edges
            .filter((e) => e.target === sid && idIsType(e.source, 'feedingTermNode'))
            .map((e) => e.source);
          const fts: any[] = [];
          for (const fid of ftIds) {
            const ftState = (feedingTerms[fid] || { name: 'feeding', inputs: {}, outputs: {} }) as any;
            const metDict: Record<string, [number, number]> = {};
            // Inputs: consumption (yield positive, MonodK>0)
            Object.entries(ftState.inputs || {}).forEach(([metId, v]) => {
              const name = metNameById(metId);
              const y = Number((v as any).yield || 0) || 0;
              const K = Number((v as any).monodK || 0) || 0;
              if (!Number.isFinite(y) || !Number.isFinite(K)) return;
              metDict[name] = [y, K > 0 ? K : 0.1];
            });
            // Outputs: production (yield negative, K = 0)
            Object.entries(ftState.outputs || {}).forEach(([metId, v]) => {
              const name = metNameById(metId);
              const y = Number((v as any).yield || 0) || 0;
              if (!Number.isFinite(y)) return;
              metDict[name] = [-(Math.abs(y)), 0];
            });
            const rf = rfById(fid);
            const id = ((rf?.data as any)?.name || fid) as string;
            fts.push({ id, metDict });
          }
          subpops[spName].feedingTerms = fts;
        }

        // Build transitions via transition nodes: sp -> tn -> sp
        const transitions: Record<string, Array<[string, string, number]>> = {};
        const trNodes = rfNodes.filter((n) => n.type === 'transitionNode');
        const subpopIdsOfBac = new Set(spIds);
        for (const tn of trNodes) {
          const incoming = edges.find((e) => e.target === tn.id && subpopIdsOfBac.has(e.source));
          const outgoing = edges.find((e) => e.source === tn.id && subpopIdsOfBac.has(e.target));
          if (!incoming || !outgoing) continue;
          const srcName = spNameById[incoming.source];
          const tgtName = spNameById[outgoing.target];
          if (!srcName || !tgtName) continue;
          const td: any = tn.data || {};
          const rate = Number(td.rate || 0) || 0;
          const condition = String(td.condition || '');
          (transitions[srcName] ||= []).push([tgtName, condition, rate]);
        }

        microbiome.bacteria[species] = {
          species,
          color,
          subpopulations: subpops,
          connections: transitions,
        };
      }

      model.microbiome = microbiome;
    }

    // Emit file
    const json = JSON.stringify(model, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `model_export_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // Save all metabolites in canvas and push initials to Simulation
  const saveAllMetabolites = async () => {
    if (savingMets) return;
    try {
      setSavingMets(true);
      // Save all unsaved metabolites
      const unsaved = nodes.filter((n) => !n.saved);
      const results = await Promise.all(
        unsaved.map(async (n) => {
          try {
            const saved = await createMetabolite(n.data);
            return { id: n.localId, saved } as const;
          } catch (e) {
            console.error('Failed to save metabolite', n.data?.name, e);
            return null;
          }
        })
      );
      const okResults = results.filter(Boolean) as Array<{ id: string; saved: MetaboliteOut }>;
      if (okResults.length > 0) {
        setNodes((prev) =>
          prev.map((n) => {
            const r = okResults.find((x) => x.id === n.localId);
            return r ? { ...n, saved: true, serverId: r.saved.id, lastSaved: r.saved } : n;
          })
        );
        setRfNodes((prev) =>
          prev.map((rf) => {
            const r = okResults.find((x) => x.id === rf.id);
            return r ? { ...rf, data: { ...(rf.data as any), saved: true } } : rf;
          })
        );
      }

      // Build initials from current canvas metabolites
      const initials: Record<string, number> = {};
      for (const rf of rfNodes) {
        if (rf.type === 'metaboliteNode') {
          const d: any = rf.data || {};
          const name = String(d.name || rf.id);
          const conc = Number(d.concentration ?? 0) || 0;
          initials[name] = conc;
        }
      }
      try {
        window.dispatchEvent(new CustomEvent('metabolites:initials', { detail: { initials } }));
      } catch (e) {
        console.error('Failed to dispatch metabolites:initials', e);
      }
      alert(`Saved ${okResults.length} new molecule${okResults.length === 1 ? '' : 's'} and updated Simulation initials.`);
    } finally {
      setSavingMets(false);
    }
  };

  const deleteAllSaved = async () => {
    const ok = window.confirm(
      "Delete ALL saved objects? This removes microbiomes, bacteria, timelines, pulses, environments, pH functions, metabolomes, and metabolites."
    );
    if (!ok) return;
    setNuking(true);
    try {
      const [micros, bac, sims, tls, pulses, envs, phs, phDrafts, metasomes, mets] = await Promise.all([
        listMicrobiomes().catch(() => []),
        listBacteria().catch(() => []),
        listSimulations().catch(() => []),
        listTimelines().catch(() => []),
        listPulses().catch(() => []),
        listEnvironments().catch(() => []),
        listPHFunctions().catch(() => []),
        listPHDrafts().catch(() => []),
        listMetabolomes().catch(() => []),
        listMetabolites().catch(() => []),
      ]);
      // Delete in dependency-safe order
      await Promise.allSettled(micros.map((m) => deleteMicrobiome(m.id)));
      await Promise.allSettled(bac.map((b) => deleteBacteria(b.id)));
      await Promise.allSettled(sims.map((s) => deleteSimulation(s.id)));
      await Promise.allSettled(tls.map((t) => deleteTimeline(t.id)));
      await Promise.allSettled(pulses.map((p) => deletePulse(p.id)));
      await Promise.allSettled(envs.map((e) => deleteEnvironment(e.id)));
      await Promise.allSettled(phs.map((f) => deletePHFunction(f.id)));
      await Promise.allSettled(phDrafts.map((d: any) => deletePHDraft(d.id)));
      await Promise.allSettled(metasomes.map((m) => deleteMetabolome(m.id)));
      await Promise.allSettled(mets.map((m) => deleteMetabolite(m.id)));

      // Clear local drafts/snapshots
      try { localStorage.removeItem('mg-build-ph-drafts'); } catch {}
      try { localStorage.removeItem('mg-sim-timelines'); } catch {}

      // Reset Build panel state
      setNodes([]);
      setRfNodes([]);
      setPh(null);
      setSelectedId(null);
      setSavedAt(Date.now());
      // notify other pages/tabs
      try { window.dispatchEvent(new CustomEvent('mg:dataReset')); } catch {}
      try { window.dispatchEvent(new CustomEvent('bacteria:changed')); } catch {}
    } finally {
      setNuking(false);
    }
  };

  const selected = nodes.find((n) => n.localId === selectedId) || null;
  const selectedRf = rfNodes.find((n) => n.id === selectedId) || null;

  // Load existing metabolites into canvas
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await listMetabolites();
        if (cancelled) return;
        const mapped: Node[] = list.map((m, i) => ({
          localId: `server-${m.id}`,
          serverId: m.id,
          saved: true,
          x: 40 + (i % 3) * 180,
          y: 60 + Math.floor(i / 3) * 120,
          data: {
            name: m.name,
            concentration: m.concentration,
            formula: m.formula,
            color: m.color,
            description: m.description,
          },
          lastSaved: m,
        }));
        setNodes(mapped);
        setRfNodes(
          mapped.map<RFNode>((n) => ({
            id: n.localId,
            type: "metaboliteNode",
            position: { x: n.x, y: n.y },
            data: { name: n.data.name, concentration: n.data.concentration, color: n.data.color, saved: n.saved },
          }))
        );
      } catch {
        // ignore for now
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load available presets
  React.useEffect(() => {
    (async () => {
      try {
        const list = await listPresets();
        setPresets(list.map((p) => ({ id: p.id, name: p.name })));
      } catch {}
    })();
  }, []);

  const loadPreset = async (presetId: string) => {
    const data = await getPreset(presetId);
    if (!data) {
      alert("Failed to load preset");
      return;
    }
    // Clear existing
    setNodes([]);
    setRfNodes([]);
    setPh(null);
    setEdges([]);
    setInitialEdges([]);

    // Layout helpers
    const nextPos = (() => {
      let y = 60;
      return (x: number) => { const p = { x, y }; y += 110; return p; };
    })();

    const idByMetName = new Map<string, string>();
    const rfList: RFNode[] = [];
    const newEdges: Array<{ id: string; source: string; target: string }> = [];

    // Metabolites
    const mets = (data?.metabolome?.metabolites || []) as any[];
    const newNodes: Node[] = [];
    mets.forEach((m: any, i: number) => {
      const localId = `m-${m.name}`;
      idByMetName.set(m.name, localId);
      newNodes.push({ localId, saved: false, x: 60, y: 60 + i * 110, data: { name: m.name, concentration: m.concentration ?? 0, formula: m.formula || { C:0,H:0,O:0,N:0,S:0,P:0 }, color: m.color || "#22d3ee", description: m.description || "" } });
      rfList.push({ id: localId, type: 'metaboliteNode', position: { x: 60, y: 60 + i * 110 }, data: { name: m.name, concentration: m.concentration ?? 0, color: m.color || "#22d3ee", saved: false } } as any);
    });

    // pH
    const phData = data?.metabolome?.pH;
    if (phData) {
      const phId = 'ph-1';
      setPh({ id: phId, baseValue: Number(phData.baseValue ?? 7.0), weights: { ...(phData.connectedMetabolites || {}) }, saved: false });
      rfList.push({ id: phId, type: 'phNode', position: { x: 320, y: 60 }, data: { baseValue: Number(phData.baseValue ?? 7.0) } } as any);
    }

    // Bacteria species → subpops, feeding terms, transitions
    const bac = data?.microbiome?.bacteria || {} as Record<string, any>;
    const subpopIdByName = new Map<string, string>();
    const feedingTermsLocal: typeof feedingTerms = {};

    const subpopColumnX = 520;
    const feedTermColumnX = 360;
    const transitionColumnX = 700;
    const bacteriaColumnX = 860;
    let subpopRow = 0;

    Object.values(bac).forEach((spec: any) => {
      // Bacteria node
      const bacId = `bac-${spec.species}`;
      rfList.push({ id: bacId, type: 'bacteriaNode', position: { x: bacteriaColumnX, y: 120 + subpopRow * 120 }, data: { species: String(spec.species), color: String(spec.color || '#54f542') } } as any);

      const subpops = spec.subpopulations || {};
      Object.values(subpops).forEach((sp: any) => {
        const spId = `sp-${spec.species}-${sp.name}`;
        subpopIdByName.set(sp.name, spId);
        rfList.push({ id: spId, type: 'subpopulationNode', position: { x: subpopColumnX, y: 160 + subpopRow * 120 }, data: { name: String(sp.name), count: Number(sp.count||0), mumax: Number(sp.mumax||0), state: String(sp.state||'active'), color: String(sp.color||'#aaaaaa'), pHopt: Number(sp.pHopt||7), pH_sensitivity_left: Number(sp.pH_sensitivity_left||2), pH_sensitivity_right: Number(sp.pH_sensitivity_right||2), Topt: Number(sp.Topt||37), tempSensitivity_left: Number(sp.tempSensitivity_left||5), tempSensitivity_right: Number(sp.tempSensitivity_right||2) } } as any);
        subpopRow += 1;
        // Feeding terms for this subpop
        (sp.feedingTerms || []).forEach((ft: any, j: number) => {
          const ftId = `ft-${spec.species}-${sp.name}-${j}`;
          const metDict = ft.metDict || {};
          feedingTermsLocal[ftId] = { name: String(ft.id || 'feeding'), inputs: {}, outputs: {} } as any;
          const ftLoc = feedingTermsLocal[ftId]!;
          // Inputs: yield>0 with K>0; outputs: K==0
          Object.entries(metDict).forEach(([met, arr]) => {
            const y = Number(Array.isArray(arr) ? arr[0] : 0) || 0;
            const K = Number(Array.isArray(arr) ? arr[1] : 0) || 0;
            const metId = idByMetName.get(met) || `m-${met}`;
            if (K > 0) {
              ftLoc.inputs[metId] = { yield: y, monodK: K };
            } else {
              ftLoc.outputs[metId] = { yield: Math.abs(y) };
            }
          });
          const ftRec = feedingTermsLocal[ftId]!;
          rfList.push({ id: ftId, type: 'feedingTermNode', position: { x: feedTermColumnX, y: (160 + (subpopRow-1) * 120) }, data: { name: String(ft.id || 'feeding'), inputs: Object.keys(ftRec.inputs).length, outputs: Object.keys(ftRec.outputs).length } } as any);
          // Edges are tracked outside; inspector will show counts
        });
      });

      // Connect subpops to bacteria
      Object.values(subpops).forEach((sp: any, idx: number) => {
        const spId = `sp-${spec.species}-${sp.name}`;
        newEdges.push({ id: `${spId}->${bacId}`, source: spId, target: bacId });
      });

      // Transitions: create a transition node per pair and place in column
      const conns = (spec.connections || {}) as Record<string, any[]>;
      Object.entries(conns).forEach(([src, lst]: [string, any[]]) => {
        (lst || []).forEach((tr: any, k: number) => {
          const [tgt, condition, rate] = tr;
          const tnId = `tr-${spec.species}-${src}-${k}`;
          rfList.push({ id: tnId, type: 'transitionNode', position: { x: transitionColumnX, y: 200 + (subpopRow + k) * 80 }, data: { rate: Number(rate||0), condition: String(condition||'') } } as any);
          const srcId = `sp-${spec.species}-${src}`;
          const tgtId = `sp-${spec.species}-${tgt}`;
          newEdges.push({ id: `${srcId}->${tnId}`, source: srcId, target: tnId });
          newEdges.push({ id: `${tnId}->${tgtId}`, source: tnId, target: tgtId });
        });
      });
    });

    // Commit metabolites and RF nodes
    setNodes(newNodes);
    setRfNodes(rfList);

    // Build edges for pH and feeding terms after nodes exist
    if (phData && phData.connectedMetabolites) {
      Object.keys(phData.connectedMetabolites).forEach((met) => {
        const srcId = idByMetName.get(met) || `m-${met}`;
        newEdges.push({ id: `${srcId}->ph-1`, source: srcId, target: 'ph-1' });
      });
    }
    // Feed term edges: metabolite→ft and ft→metabolite, and ft→subpop
    Object.entries(feedingTermsLocal).forEach(([ftId, ft]) => {
      Object.keys(ft.inputs || {}).forEach((metLocalId) => {
        newEdges.push({ id: `${metLocalId}->${ftId}`, source: metLocalId, target: ftId });
      });
      Object.keys(ft.outputs || {}).forEach((metLocalId) => {
        newEdges.push({ id: `${ftId}->${metLocalId}`, source: ftId, target: metLocalId });
      });
      // Find owning subpopulation by id pattern
      const parts = ftId.split('-');
      const species = parts[1];
      const subpop = parts[2];
      const spId = `sp-${species}-${subpop}`;
      newEdges.push({ id: `${ftId}->${spId}`, source: ftId, target: spId });
    });

    // Sync feedingTerms state for inspector
    setFeedingTerms(feedingTermsLocal);

    // Commit edges (for both canvas and validation)
    setInitialEdges(newEdges);
    setEdges(newEdges);
  };

  const addNode = () => {
    const id = `local-${Math.random().toString(36).slice(2, 8)}`;
    // Ensure unique default metabolite name
    const existing = new Set(nodes.map((n) => n.data.name));
    let base = 'metabolite'; let name = base; let i = 1; while (existing.has(name)) name = `${base}_${i++}`;
    setNodes((prev) => [
      ...prev,
      { localId: id, saved: false, x: 60, y: 80, data: { ...defaultMetabolite(), name } },
    ]);
    setSelectedId(id);
    setRfNodes((prev) => [
      ...prev,
      {
        id,
        type: "metaboliteNode",
        position: { x: 60, y: 80 },
        data: { name, concentration: 1, color: "#22d3ee", saved: false },
      },
    ]);
    setCenterNodeId(id);
  };

  // Add a single pH node
  const addPhNode = () => {
    if (ph && !ph.saved) return; // block when unsaved exists
    const id = 'ph-1';
    // Remove previous pH rf node if existed and was saved
    if (ph && ph.saved) {
      setRfNodes((prev) => prev.filter((n) => n.id !== ph.id));
    }
    setPh({ id, baseValue: 7.0, weights: {}, saved: false });
    setRfNodes((prev) => [
      ...prev,
      { id, type: 'phNode', position: { x: 300, y: 60 }, data: { baseValue: 7.0 } as any },
    ]);
    setSelectedId(id);
    setCenterNodeId(id);
    setShowAddMenu(false);
  };

  // Add Feeding Term node
  const addFeedingTermNode = () => {
    const id = `ft-${Math.random().toString(36).slice(2, 8)}`;
    const existing = new Set(
      rfNodes.filter((n) => n.type === 'feedingTermNode').map((n) => String(((n.data as any)?.name || 'feeding')))
    );
    let base = 'feeding', name = base, i = 1; while (existing.has(name)) name = `${base}_${i++}`;
    setRfNodes((prev) => [
      ...prev,
      { id, type: 'feedingTermNode', position: { x: 360, y: 160 }, data: { name, inputs: 0, outputs: 0 } as any },
    ]);
    setSelectedId(id);
    setCenterNodeId(id);
    setShowAddMenu(false);
  };

  // Add Subpopulation node
  const addSubpopulationNode = () => {
    const id = `sp-${Math.random().toString(36).slice(2, 8)}`;
    // Generate a unique default name among existing subpopulation nodes
    const existingNames = new Set(
      rfNodes
        .filter((n) => n.type === 'subpopulationNode')
        .map((n) => String(((n.data as any)?.name || 'subpop')).trim())
    );
    const base = 'subpop';
    let name = base;
    let i = 1;
    while (existingNames.has(name)) {
      name = `${base}_${i++}`;
    }
    setRfNodes((prev) => [
      ...prev,
      { id, type: 'subpopulationNode', position: { x: 520, y: 180 }, data: { name, mumax: 0.5, state: 'active' } as any },
    ]);
    setSelectedId(id);
    setCenterNodeId(id);
    setShowAddMenu(false);
  };

  // Add Transition node
  const addTransitionNode = () => {
    const id = `tr-${Math.random().toString(36).slice(2, 8)}`;
    setRfNodes((prev) => [
      ...prev,
      { id, type: 'transitionNode', position: { x: 700, y: 220 }, data: { rate: 0.1, condition: '' } as any },
    ]);
    setSelectedId(id);
    setCenterNodeId(id);
    setShowAddMenu(false);
  };

  // Add Bacteria node
  const addBacteriaNode = () => {
    const id = `bac-${Math.random().toString(36).slice(2, 8)}`;
    const existing = new Set(
      rfNodes.filter((n) => n.type === 'bacteriaNode').map((n) => String(((n.data as any)?.species || 'species')))
    );
    let base = 'species', species = base, i = 1; while (existing.has(species)) species = `${base}_${i++}`;
    setRfNodes((prev) => [
      ...prev,
      { id, type: 'bacteriaNode', position: { x: 860, y: 120 }, data: { species, color: '#54f542' } as any },
    ]);
    setSelectedId(id);
    setCenterNodeId(id);
    setShowAddMenu(false);
  };

  // Save all bacteria nodes
  const saveAllBacteria = async () => {
    try {
      const bacteriaNodes = rfNodes.filter((n) => n.type === 'bacteriaNode');
      if (bacteriaNodes.length === 0) {
        alert('No bacteria nodes to save');
        return;
      }

      let savedCount = 0;
      for (const bacteriaNode of bacteriaNodes) {
        const species = (bacteriaNode.data as any).species || 'species';
        const color = (bacteriaNode.data as any).color || '#54f542';
        
        // Collect subpopulations connected to this bacteria node
        const subpopIds = edges.filter((e) => e.target === bacteriaNode.id)
          .map((e) => e.source)
          .filter((id) => (rfNodes.find((n) => n.id === id)?.type === 'subpopulationNode'));
        
        const subpopulations: SubpopulationIn[] = subpopIds.map((sid) => {
          const sn = rfNodes.find((n) => n.id === sid)!;
          const d = sn.data as any;
          return {
            name: String(d.name || sid),
            species,
            count: Number(d.count || 0) || 0,
            mumax: Number(d.mumax || 0) || 0,
            feedingTerms: [],
            pHopt: Number(d.pHopt || 7.0) || 7.0,
            pH_sensitivity_left: Number(d.pH_sensitivity_left || 2.0) || 2.0,
            pH_sensitivity_right: Number(d.pH_sensitivity_right || 2.0) || 2.0,
            Topt: Number(d.Topt || 37.0) || 37.0,
            tempSensitivity_left: Number(d.tempSensitivity_left || 5.0) || 5.0,
            tempSensitivity_right: Number(d.tempSensitivity_right || 2.0) || 2.0,
            state: String(d.state || 'active'),
            color: String(d.color || '#aaaaaa'),
          } as SubpopulationIn;
        });

        // Map id -> name for metabolite nodes
        const metNameById = (id: string) => {
          const match = nodes.find((n) => n.localId === id);
          return match ? match.data.name : id;
        };

        // Attach feeding terms per subpopulation: edges feedingTermNode -> subpopulationNode
        for (const sp of subpopulations) {
          const spRf = rfNodes.find((n) => n.type === 'subpopulationNode' && String((n.data as any)?.name || '') === sp.name);
          const spId = spRf?.id;
          const ftIds = edges
            .filter((e) => e.target === spId && (rfNodes.find((n) => n.id === e.source)?.type === 'feedingTermNode'))
            .map((e) => e.source);
          const fts: FeedingTermIn[] = ftIds.map((fid) => {
            const ftState = (feedingTerms[fid] || { name: 'feeding', inputs: {}, outputs: {} }) as any;
            const metDict: Record<string, [number, number]> = {};
            Object.entries(ftState.inputs || {}).forEach(([metId, v]) => {
              const name = metNameById(metId);
              const y = Number((v as any).yield || 0) || 0;
              const K = Number((v as any).monodK || 0) || 0;
              if (!Number.isFinite(y) || !Number.isFinite(K)) return;
              metDict[name] = [y, K > 0 ? K : 0.1];
            });
            Object.entries(ftState.outputs || {}).forEach(([metId, v]) => {
              const name = metNameById(metId);
              const y = Number((v as any).yield || 0) || 0;
              if (!Number.isFinite(y)) return;
              metDict[name] = [-(Math.abs(y)), 0];
            });
            const rf = rfNodes.find((n) => n.id === fid);
            const id = (rf?.data as any)?.name || fid;
            return { id, metDict } as FeedingTermIn;
          });
          (sp as any).feedingTerms = fts;
        }

        // Transitions via transition nodes (restricted to this bacteria's subpopulations)
        const transitions = rfNodes.filter((n) => n.type === 'transitionNode');
        const connections: Record<string, TransitionIn[]> = {};
        const subpopIdsSet = new Set(subpopIds);
        function subpopNameById(id: string): string | null {
          const sp = rfNodes.find((n) => n.id === id && n.type === 'subpopulationNode');
          if (!sp) return null;
          return String((sp.data as any)?.name || id);
        }
        for (const tn of transitions) {
          const incoming = edges.find((e) => e.target === tn.id && subpopIdsSet.has(e.source));
          const outgoing = edges.find((e) => e.source === tn.id && subpopIdsSet.has(e.target));
          if (!incoming || !outgoing) continue;
          const srcName = subpopNameById(incoming.source);
          const tgtName = subpopNameById(outgoing.target);
          if (!srcName || !tgtName) continue;
          const rate = Number((tn.data as any)?.rate || 0) || 0;
          const condition = String((tn.data as any)?.condition || '');
          if (!connections[srcName]) connections[srcName] = [];
          connections[srcName].push({ target: tgtName, condition, rate });
        }

        const payload: BacteriaIn = { species, color, subpopulations, connections };
        console.log('Saving bacteria with payload:', payload);
        const saved = await createBacteria(payload);
        console.log('Bacteria saved successfully:', saved);
        savedCount++;
      }

      try { 
        console.log('Dispatching bacteria:changed event');
        window.dispatchEvent(new CustomEvent('bacteria:changed')); 
      } catch (e) {
        console.error('Failed to dispatch bacteria:changed event:', e);
      }
      
      alert(`Saved ${savedCount} microbes`);
    } catch (e) {
      console.error(e);
      const msg = e instanceof Error ? e.message : String(e);
      alert(`Failed to save bacteria: ${msg}`);
    }
  };

  // Add a predefined metabolite from the quick menu
  const addPredefined = (m: { name: string; color: string; concentration: number; unit: string; formula: Record<string, number> }) => {
    const id = `local-${Math.random().toString(36).slice(2, 8)}`;
    const existing = new Set(nodes.map((n) => n.data.name));
    let base = m.name.toLowerCase();
    let name = base; let i = 1; while (existing.has(name)) name = `${base}_${i++}`;
    const data: MetaboliteIn = {
      name,
      concentration: m.concentration,
      formula: m.formula as any,
      color: m.color,
      description: `${m.name} metabolite`,
    };

    setNodes((prev) => [
      ...prev,
      { localId: id, saved: false, x: 80, y: 100, data },
    ]);
    setRfNodes((prev) => [
      ...prev,
      { id, type: 'metaboliteNode', position: { x: 80, y: 100 }, data: { name: data.name, concentration: data.concentration, color: data.color, saved: false } },
    ]);
    setSelectedId(id);
    setCenterNodeId(id);
    setShowAddMenu(false);
    setShowMetaboliteSub(false);
  };

  const updateSelected = (patch: Partial<MetaboliteIn>) => {
    if (!selected) return;
    const prevName = selected.data.name;
    setNodes((prev) =>
      prev.map((n) =>
        n.localId === selected.localId ? { ...n, data: { ...n.data, ...patch } } : n,
      ),
    );
    setRfNodes((prev) =>
      prev.map((n) =>
        n.id === selected.localId
          ? {
              ...n,
              data: {
                ...(n.data as any),
                name: patch.name ?? (n.data as any).name,
                concentration:
                  (patch.concentration as number | undefined) ?? (n.data as any).concentration,
                color: patch.color ?? (n.data as any).color,
              },
            }
          : n,
      ),
    );
    if (patch.name && ph && prevName !== patch.name) {
      setPh((prev) => {
        if (!prev) return prev;
        if (Object.prototype.hasOwnProperty.call(prev.weights, prevName)) {
          const w = Number(prev.weights[prevName]);
          const nextW = { ...prev.weights } as Record<string, number>;
          delete nextW[prevName];
          nextW[patch.name as string] = w;
          return { ...prev, weights: nextW };
        }
        return prev;
      });
    }
  };

  const saveSelected = async () => {
    if (!selected) return;
    try {
      setSaving(true);
      const saved = await createMetabolite(selected.data);
      setNodes((prev) =>
        prev.map((n) =>
          n.localId === selected.localId
            ? { ...n, saved: true, serverId: saved.id, lastSaved: saved }
            : n,
        ),
      );
      setRfNodes((prev) =>
        prev.map((n) =>
          n.id === selected.localId ? { ...n, data: { ...(n.data as any), saved: true } } : n,
        ),
      );
      setSavedAt(Date.now());
      setSelectedId(null); // unselect after successful save
    } catch (e) {
      console.error(e);
      alert("Failed to save metabolite");
    } finally { setSaving(false); }
  };

  const deleteSelected = async () => {
    if (!selected) return;
    if (selected.serverId) {
      try {
        await deleteMetabolite(selected.serverId);
      } catch {
        /* ignore */
      }
    }
    setNodes((prev) => prev.filter((n) => n.localId !== selected.localId));
    setRfNodes((prev) => prev.filter((n) => n.id !== selected.localId));
    setSelectedId(null);
  };

  const clearAll = async () => {
    if (nodes.length === 0) return;
    const ok = window.confirm("Delete all nodes? This will also remove any saved metabolites.");
    if (!ok) return;
    setClearing(true);
    try {
      const ids = nodes.map((n) => n.serverId).filter(Boolean) as string[];
      await Promise.all(ids.map((id) => deleteMetabolite(id).catch(() => {})));
      setNodes([]);
      setRfNodes([]);
      setPh(null);
      setSelectedId(null);
    } finally {
      setClearing(false);
    }
  };

  const FormulaEditor = ({ value, onChange }: { value: MetaboliteIn["formula"]; onChange: (f: MetaboliteIn["formula"]) => void; }) => {
    const keys: Array<keyof MetaboliteIn["formula"]> = ["C", "H", "O", "N", "S", "P"];
    return (
      <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8 }}>
        {keys.map((k) => (
          <label key={k} style={{ display: "grid", gap: 4 }}>
            <span className="label">{k}</span>
            <input
              type="number"
              min={0}
              step={1}
              value={value[k] ?? 0}
              onChange={(e) => onChange({ ...value, [k]: Number(e.target.value) })}
              className="input"
            />
          </label>
        ))}
      </div>
    );
  };

  // Validation for inspector
  const errors = React.useMemo(() => {
    const e: Record<string, string> = {};
    if (!selected) return e;
    if (!selected.data.name.trim()) e.name = "Name is required";
    // Enforce unique metabolite names in canvas (by name)
    const duplicate = nodes.some((n) => n.localId !== selected.localId && n.data.name.trim() === selected.data.name.trim());
    if (duplicate) e.name = "Name must be unique";
    if (!(selected.data.concentration >= 0)) e.concentration = "Must be â‰¥ 0";
    const f = selected.data.formula;
    const keys: (keyof MetaboliteIn["formula"])[] = ["C","H","O","N","S","P"];
    for (const k of keys) {
      const v = f[k] ?? 0;
      if (!(Number.isInteger(v) && v >= 0)) { e.formula = "Elements must be non-negative integers"; break; }
    }
    return e;
  }, [selected]);

  return (
    <div className="grid" id="panel-build">
      <div className="card" style={{ gridColumn: "span 12" }}>
        <div className="row space-between center" style={{ position: 'relative' }}>
          <ApiStatus />
          <div className="row" style={{ gap: 8, alignItems: 'center' }}>
            <button className="btn" onClick={deleteAllSaved} type="button" title="Delete ALL saved objects" disabled={nuking}>
              {nuking ? 'Deletingâ€¦' : 'Delete all'}
            </button>
            <button className="btn ghost" onClick={handleExportJson} type="button" title="Export model as JSON">
              Export JSON
            </button>
            
            <button
              className="btn primary btn--icon"
              onClick={() => { setShowAddMenu((s) => !s); if (showMetaboliteSub) setShowMetaboliteSub(false); }}
              type="button"
              title="Add"
              aria-haspopup="menu"
            >
              +
            </button>
            <button
              className="btn primary"
              onClick={saveAllMetabolites}
              type="button"
              title="Save all molecules and send to Simulation"
              disabled={savingMets}
            >
              {savingMets ? 'Saving molecules...' : 'Save molecules'}
            </button>
            <button
              className="btn primary"
              onClick={async () => {
                if (!ph) { alert('No pH node to save'); return; }
                const name = prompt('Name for this pH draft?', 'pH Draft');
                if (!name) return;
                try {
                  const weights: Record<string, number> = {};
                  Object.entries(ph.weights || {}).forEach(([k, v]) => {
                    const f = Number(v);
                    if (Number.isFinite(f)) weights[k] = f;
                  });
                  const base = Math.max(0, Math.min(14, Number(ph.baseValue)));
                  await createPHDraft({ name, baseValue: base, weights });
                  try { window.dispatchEvent(new CustomEvent('ph:draftSaved')); } catch {}
                  alert('Saved pH draft');
                } catch (e) {
                  console.error(e);
                  alert('Failed to save pH draft');
                }
              }}
              type="button"
              title="Save pH as draft (metabolome-agnostic)"
              disabled={!ph || !(Number.isFinite(ph.baseValue) && ph.baseValue >= 0 && ph.baseValue <= 14)}
              style={{ marginLeft: 8 }}
            >
              Save pH
            </button>
            <button
              className="btn primary"
              onClick={saveAllBacteria}
              type="button"
              title="Save all microbes"
              style={{ marginLeft: 8 }}
            >
              Save microbes
            </button>
            {showAddMenu && (
              <div
                role="menu"
                style={{
                  position: 'absolute',
                  right: 0,
                  top: 'calc(100% + 8px)',
                  background: 'color-mix(in oklab, var(--bg-elev), white 12%)',
                  border: '1px solid var(--border)',
                  borderRadius: 10,
                  boxShadow: 'var(--shadow)',
                  minWidth: 220,
                  padding: 8,
                  zIndex: 20,
                  maxHeight: 320,
                  overflow: 'auto',
                }}
              >
                <button className="btn" style={{ width: '100%' }} onClick={addPhNode} disabled={!!ph && !ph.saved}>
                  pH {ph ? (ph.saved ? '(savedâ€"click to replace)' : '(unsaved)') : ''}
                </button>
                {!showMetaboliteSub ? (
                  <button className="btn" style={{ width: '100%' }} onClick={() => setShowMetaboliteSub(true)}>
                    Metabolite {'»'}
                  </button>
                ) : (
                  <div style={{ display: 'grid', gap: 6 }}>
                    <button className="btn" onClick={() => setShowMetaboliteSub(false)}>{'«'} Back</button>
                    {predefinedMetabolites.map((m) => (
                      <button
                        key={m.name}
                        className="btn"
                        onClick={() => addPredefined(m)}
                        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                      >
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ width: 10, height: 10, borderRadius: 999, background: m.color }} />
                          {m.name}
                        </span>
                        <span className="muted" style={{ fontSize: 12 }}>{m.concentration} {m.unit}</span>
                      </button>
                    ))}
                    <button className="btn" onClick={addNode}>+ New metabolite</button>
                  </div>
                )}
                <hr style={{ margin: '8px 0', borderColor: 'var(--border)', opacity: 0.4 }} />
                <button className="btn" style={{ width: '100%' }} onClick={addFeedingTermNode}>Feeding term</button>
                <button className="btn" style={{ width: '100%' }} onClick={addSubpopulationNode}>Subpopulation</button>
                <button className="btn" style={{ width: '100%' }} onClick={addTransitionNode}>Transition</button>
                <button className="btn" style={{ width: '100%' }} onClick={addBacteriaNode}>Microbial Species</button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Canvas using React Flow */}
      <div className="card" style={{ gridColumn: "span 8" }}>
        <h2>Canvas</h2>
        <p className="muted">Drag nodes, click to edit in the inspector.</p>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <label className="label" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }} title="Canvas color">
            <span className="muted">Canvas</span>
            <input
              type="color"
              value={canvasBg}
              onChange={(e) => setCanvasBg(e.target.value)}
              className="input"
              style={{ width: 44, height: 34, padding: 4 }}
              aria-label="Canvas color"
            />
          </label>
          <button className="btn" onClick={clearAll} disabled={clearing || nodes.length === 0}>
            {clearing ? "Clearing..." : "Clear all"}
          </button>
        </div>
        <ReactFlowProvider>
          <BuildCanvas
            nodes={rfNodes}
            initialEdges={initialEdges}
            centerOnNodeId={centerNodeId}
            backgroundColor={canvasBg}
            gridColor={gridFor(canvasBg)}
            onNodesUpdate={(rfNodes) => {
            // accept updates from React Flow (positions, selection flags)
              setRfNodes(rfNodes);
            // sync positions back to our plain nodes
              setNodes((prev) =>
                prev.map((n) => {
                  const rf = rfNodes.find((r) => r.id === n.localId);
                  return rf ? { ...n, x: rf.position.x, y: rf.position.y } : n;
                }),
              );
            }}
            onSelect={(id) => setSelectedId(id)}
            onEdgeCreate={({ source, target }) => {
            // Track edges for bacteria/subpopulation/transition topology
              setEdges((prev) => (prev.some((e) => e.source === source && e.target === target) ? prev : [...prev, { id: `${source}->${target}`, source, target }]));
              const src = rfNodes.find((n) => n.id === source);
              const tgt = rfNodes.find((n) => n.id === target);
              if (!src || !tgt) return;
              const s = String(src.type);
              const t = String(tgt.type);
            // metabolite -> pH (existing)
              if (t === 'phNode' && s === 'metaboliteNode' && ph) {
                const metName = (src.data as any).name as string;
                setPh((prev) => {
                  if (!prev) return prev;
                  if (Object.prototype.hasOwnProperty.call(prev.weights, metName)) return prev;
                  return { ...prev, weights: { ...prev.weights, [metName]: -1 } };
                });
                return;
              }
            // metabolite -> feedingTerm (input)
              if (s === 'metaboliteNode' && t === 'feedingTermNode') {
              const ftId = target;
              const metId = source;
              setFeedingTerms((prev) => {
                const cur = prev[ftId] || { name: 'feeding', inputs: {}, outputs: {} };
                if (cur.inputs[metId]) return prev; // already connected
                const nextFt = { ...cur, inputs: { ...cur.inputs, [metId]: { yield: 1.0, monodK: 0.5 } } };
                const out = { ...prev, [ftId]: nextFt };
                // update RF node counts
                setRfNodes((rfs) => rfs.map((n) => n.id === ftId ? { ...n, data: { ...(n.data as any), inputs: Object.keys(nextFt.inputs).length, outputs: Object.keys(nextFt.outputs).length } } : n));
                return out;
              });
              return;
              }
            // feedingTerm -> metabolite (output)
              if (s === 'feedingTermNode' && t === 'metaboliteNode') {
              const ftId = source;
              const metId = target;
              setFeedingTerms((prev) => {
                const cur = prev[ftId] || { name: 'feeding', inputs: {}, outputs: {} };
                if (cur.outputs[metId]) return prev;
                const nextFt = { ...cur, outputs: { ...cur.outputs, [metId]: { yield: 1.0 } } };
                const out = { ...prev, [ftId]: nextFt };
                setRfNodes((rfs) => rfs.map((n) => n.id === ftId ? { ...n, data: { ...(n.data as any), inputs: Object.keys(nextFt.inputs).length, outputs: Object.keys(nextFt.outputs).length } } : n));
                return out;
              });
              return;
              }
            }}
            onEdgeRemove={({ source, target }) => {
            const src = rfNodes.find((n) => n.id === source);
            const tgt = rfNodes.find((n) => n.id === target);
            if (!src || !tgt) return;
            const s = String(src.type);
            const t = String(tgt.type);
            // Drop from edges snapshot
            setEdges((prev) => prev.filter((e) => !(e.source === source && e.target === target)));
            // metabolite -> pH detach
            if (t === 'phNode' && s === 'metaboliteNode' && ph) {
              const metName = (src.data as any).name as string;
              setPh((prev) => {
                if (!prev) return prev;
                if (!Object.prototype.hasOwnProperty.call(prev.weights, metName)) return prev;
                const nextW = { ...prev.weights } as Record<string, number>;
                delete nextW[metName];
                return { ...prev, weights: nextW };
              });
              return;
            }
            // metabolite -> feedingTerm (input) detach
            if (s === 'metaboliteNode' && t === 'feedingTermNode') {
              const ftId = target;
              const metId = source;
              setFeedingTerms((prev) => {
                const cur = prev[ftId];
                if (!cur || !cur.inputs[metId]) return prev;
                const nextInputs = { ...cur.inputs } as any;
                delete nextInputs[metId];
                const nextFt = { ...cur, inputs: nextInputs };
                const out = { ...prev, [ftId]: nextFt };
                setRfNodes((rfs) => rfs.map((n) => n.id === ftId ? { ...n, data: { ...(n.data as any), inputs: Object.keys(nextFt.inputs).length, outputs: Object.keys(nextFt.outputs).length } } : n));
                return out;
              });
              return;
            }
            // feedingTerm -> metabolite (output) detach
            if (s === 'feedingTermNode' && t === 'metaboliteNode') {
              const ftId = source;
              const metId = target;
              setFeedingTerms((prev) => {
                const cur = prev[ftId];
                if (!cur || !cur.outputs[metId]) return prev;
                const nextOutputs = { ...cur.outputs } as any;
                delete nextOutputs[metId];
                const nextFt = { ...cur, outputs: nextOutputs };
                const out = { ...prev, [ftId]: nextFt };
                setRfNodes((rfs) => rfs.map((n) => n.id === ftId ? { ...n, data: { ...(n.data as any), inputs: Object.keys(nextFt.inputs).length, outputs: Object.keys(nextFt.outputs).length } } : n));
                return out;
              });
              return;
            }
            }}
          />
        </ReactFlowProvider>
      </div>

      {/* Inspector */}
      <div
        className="card"
        style={{
          gridColumn: "span 4",
          overflow: "hidden",
          padding:  "12px 16px",
          background: "color-mix(in oklab, var(--panel), white 14%)",
          border: "1px solid color-mix(in oklab, var(--border), white 25%)",
        }}
      >
        <>
        {ph && selectedId === ph.id && (
          <PHInspector
            ph={ph}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
            setPh={setPh}
            setRfNodes={setRfNodes}
          />
        )}
        {selectedRf && selectedRf.type === 'feedingTermNode' && (
          <FeedingTermInspector
            selectedRf={selectedRf}
            nodes={nodes}
            feedingTerms={feedingTerms}
            setFeedingTerms={setFeedingTerms}
            setRfNodes={setRfNodes}
          />
        )}
        {selectedRf && selectedRf.type === 'subpopulationNode' && (
          <SubpopulationInspector
            selectedRf={selectedRf}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
            setRfNodes={setRfNodes}
          />
        )}
        {selectedRf && selectedRf.type === 'transitionNode' && (
          <TransitionInspector
            selectedRf={selectedRf}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
            setRfNodes={setRfNodes}
          />
        )}
        {selectedRf && selectedRf.type === 'bacteriaNode' && (
          <BacteriaInspector
            selectedRf={selectedRf}
            selectedId={selectedId}
            setSelectedId={setSelectedId}
            setRfNodes={setRfNodes}
            rfNodes={rfNodes}
            edges={edges}
            feedingTerms={feedingTerms}
          />
        )}
        
        {(!selectedRf || (selectedRf.type !== 'feedingTermNode' && selectedRf.type !== 'subpopulationNode' && selectedRf.type !== 'transitionNode' && selectedRf.type !== 'bacteriaNode')) && (
          <MetaboliteInspector
            selected={selected}
            errors={errors}
            saving={saving}
            savedAt={savedAt}
            updateSelected={updateSelected}
            saveSelected={saveSelected}
            deleteSelected={deleteSelected}
          />
        )}
        </>
      </div>

      {/* Presets placeholder (kept) */}
      <div className="card" style={{ gridColumn: "span 12" }}>
        <h3>Presets</h3>
        <p className="muted">Load a preset model into the canvas.</p>
        <div className="chip-row">
          {presets.length === 0 ? (
            <span className="muted">No presets found</span>
          ) : (
            presets.map((p) => (
              <button key={p.id} className="chip" onClick={() => void loadPreset(p.id)} title={p.name}>
                {p.name}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
