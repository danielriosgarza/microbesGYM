import React from "react";
import type { RFNode } from "../../../components/BuildCanvas";

export function BacteriaInspector({
  selectedRf,
  selectedId,
  setSelectedId,
  setRfNodes,
  rfNodes,
  edges,
  feedingTerms,
}: {
  selectedRf: RFNode;
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  setRfNodes: React.Dispatch<React.SetStateAction<RFNode[]>>;
  rfNodes: RFNode[];
  edges: Array<{ id: string; source: string; target: string }>;
  feedingTerms: Record<string, { name: string; inputs: Record<string, { yield: number; monodK: number }>; outputs: Record<string, { yield: number }> }>;
}) {
  const validationErrors = React.useMemo(() => {
    const errors: string[] = [];
    const subpopIds = edges.filter((e) => e.target === selectedRf.id)
      .map((e) => e.source)
      .filter((id) => (rfNodes.find((n) => n.id === id)?.type === 'subpopulationNode'));
    if (subpopIds.length === 0) {
      errors.push("Microbe node must have at least one connected subpopulation");
      return errors;
    }
    const subpopNames = subpopIds.map(sid => {
      const sn = rfNodes.find((n) => n.id === sid)!;
      return String((sn.data as any)?.name || sid);
    });
    const uniqueNames = new Set(subpopNames);
    if (subpopNames.length !== uniqueNames.size) {
      errors.push("Subpopulation names must be unique within a species");
    }
    for (const sid of subpopIds) {
      const sn = rfNodes.find((n) => n.id === sid)!;
      const d = sn.data as any;
      const name = String(d.name || sid);
      const state = String(d.state || 'active');
      const mumax = Number(d.mumax || 0) || 0;
      if (state === 'active') {
        if (mumax <= 0) {
          errors.push(`Active subpopulation '${name}' must have positive mumax (current: ${mumax})`);
        }
        const ftIds = edges
          .filter((e) => e.target === sid && (rfNodes.find((n) => n.id === e.source)?.type === 'feedingTermNode'))
          .map((e) => e.source);
        if (ftIds.length === 0) {
          errors.push(`Active subpopulation '${name}' must have at least one feeding term`);
        } else {
          for (const ftId of ftIds) {
            const ftState = feedingTerms[ftId];
            if (ftState) {
              const inputs = ftState.inputs || {};
              const outputs = ftState.outputs || {};
              if (Object.keys(inputs).length === 0 && Object.keys(outputs).length === 0) {
                errors.push(`Feeding term for '${name}' must have at least one input or output metabolite`);
              }
              for (const [, v] of Object.entries(inputs)) {
                const y = Number((v as any).yield || 0);
                const K = Number((v as any).monodK || 0);
                if (!Number.isFinite(y) || y <= 0) {
                  errors.push(`Input metabolite in '${name}' must have positive yield (current: ${y})`);
                }
                if (!Number.isFinite(K) || K <= 0) {
                  errors.push(`Input metabolite in '${name}' must have positive Monod K (current: ${K})`);
                }
              }
              for (const [, v] of Object.entries(outputs)) {
                const y = Number((v as any).yield || 0);
                if (!Number.isFinite(y) || y <= 0) {
                  errors.push(`Output metabolite in '${name}' must have positive yield (current: ${y})`);
                }
              }
            }
          }
        }
      }
      const count = Number(d.count || 0);
      const pHopt = Number(d.pHopt || 7.0);
      const pH_sensitivity_left = Number(d.pH_sensitivity_left || 2.0);
      const pH_sensitivity_right = Number(d.pH_sensitivity_right || 2.0);
      const Topt = Number(d.Topt || 37.0);
      const tempSensitivity_left = Number(d.tempSensitivity_left || 5.0);
      const tempSensitivity_right = Number(d.tempSensitivity_right || 2.0);
      if (count < 0) errors.push(`Subpopulation '${name}' count cannot be negative (current: ${count})`);
      if (pHopt <= 0) errors.push(`Subpopulation '${name}' pHopt must be positive (current: ${pHopt})`);
      if (pH_sensitivity_left <= 0) errors.push(`Subpopulation '${name}' pH_sensitivity_left must be positive (current: ${pH_sensitivity_left})`);
      if (pH_sensitivity_right <= 0) errors.push(`Subpopulation '${name}' pH_sensitivity_right must be positive (current: ${pH_sensitivity_right})`);
      if (Topt <= 0) errors.push(`Subpopulation '${name}' Topt must be positive (current: ${Topt})`);
      if (tempSensitivity_left <= 0) errors.push(`Subpopulation '${name}' tempSensitivity_left must be positive (current: ${tempSensitivity_left})`);
      if (tempSensitivity_right <= 0) errors.push(`Subpopulation '${name}' tempSensitivity_right must be positive (current: ${tempSensitivity_right})`);
    }
    const transitions = rfNodes.filter((n) => n.type === 'transitionNode');
    for (const tn of transitions) {
      const incoming = edges.find((e) => e.target === tn.id && (rfNodes.find((n) => n.id === e.source)?.type === 'subpopulationNode'));
      const outgoing = edges.find((e) => e.source === tn.id && (rfNodes.find((n) => n.id === e.target)?.type === 'subpopulationNode'));
      if (incoming && outgoing) {
        const subpopIdsSet = new Set(subpopIds);
        const srcName = subpopIdsSet.has(incoming.source) ? String((rfNodes.find((n) => n.id === incoming.source)?.data as any)?.name || incoming.source) : null;
        const tgtName = subpopIdsSet.has(outgoing.target) ? String((rfNodes.find((n) => n.id === outgoing.target)?.data as any)?.name || outgoing.target) : null;
        if (srcName && tgtName) {
          const rate = Number((tn.data as any)?.rate || 0);
          if (rate < 0) errors.push(`Transition rate from '${srcName}' to '${tgtName}' cannot be negative (current: ${rate})`);
        }
      }
    }
    return errors;
  }, [selectedRf, rfNodes, edges, feedingTerms]);

  return (
    <div className="form">
      <h2>Microbe</h2>
      {validationErrors.length > 0 ? (
        <div className="form-row">
          <div className="hint error" style={{ marginBottom: 16 }}>
            <strong>Validation Errors:</strong>
            <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
              {validationErrors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      <div className="form-row">
        <label className="label">Species</label>
        <input
          className="input"
          value={(selectedRf.data as any).species || 'species'}
          onChange={(e) => {
            const species = e.target.value;
            setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), species } } : n));
          }}
        />
      </div>
      <div className="form-row">
        <label className="label">Color</label>
        <div className="inline grow">
          <input
            type="color"
            className="input"
            style={{ width: 52, height: 34, padding: 4 }}
            value={(selectedRf.data as any).color || '#54f542'}
            onChange={(e) => {
              const color = e.target.value;
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), color } } : n));
            }}
          />
          <input
            className="input"
            value={(selectedRf.data as any).color || '#54f542'}
            onChange={(e) => {
              const color = e.target.value;
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), color } } : n));
            }}
            style={{ minWidth: 0 }}
          />
        </div>
      </div>
      <div className="row" style={{ gap: 8, alignItems: "center" }}>
        <button className="btn" onClick={() => {
          setRfNodes((prev) => prev.filter((n) => n.id !== selectedRf.id));
          if (selectedId === selectedRf.id) setSelectedId(null);
        }} type="button">
          Delete
        </button>
      </div>
    </div>
  );
}


