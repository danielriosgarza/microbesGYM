import React from "react";
import type { RFNode } from "../../../components/BuildCanvas";

export function FeedingTermInspector({
  selectedRf,
  nodes,
  feedingTerms,
  setFeedingTerms,
  setRfNodes,
}: {
  selectedRf: RFNode;
  nodes: Array<{ localId: string; data: { name: string } }>;
  feedingTerms: Record<string, { name: string; inputs: Record<string, { yield: number; monodK: number }>; outputs: Record<string, { yield: number }> }>;
  setFeedingTerms: React.Dispatch<React.SetStateAction<Record<string, { name: string; inputs: Record<string, { yield: number; monodK: number }>; outputs: Record<string, { yield: number }> }>>>;
  setRfNodes: React.Dispatch<React.SetStateAction<RFNode[]>>;
}) {
  const ftId = selectedRf.id;
  const ft = feedingTerms[ftId] || { name: 'feeding', inputs: {}, outputs: {} };
  // Ensure entry exists
  if (!feedingTerms[ftId]) {
    setFeedingTerms((prev) => ({ ...prev, [ftId]: ft }));
  }
  const metNameById = (id: string) => {
    const match = nodes.find((n) => n.localId === id);
    return match ? match.data.name : id;
  };
  return (
    <div className="form">
      <h2>Feeding term</h2>
      <div className="form-row">
        <label className="label">Name</label>
        <input
          className="input"
          value={ft.name}
          onChange={(e) => {
            const name = e.target.value;
            setFeedingTerms((prev) => ({ ...prev, [ftId]: { ...ft, name } }));
            setRfNodes((prev) => prev.map((n) => n.id === ftId ? { ...n, data: { ...(n.data as any), name } } : n));
          }}
        />
      </div>
      <div className="form-row">
        <label className="label">Inputs (consumed)</label>
        {Object.keys(ft.inputs).length === 0 ? (
          <div className="hint">Connect metabolite → feeding term to add inputs.</div>
        ) : (
          <div style={{ display: 'grid', gap: 6 }}>
            {Object.entries(ft.inputs).map(([mid, vals]) => (
              <div key={mid} className="inline" style={{ gap: 8, alignItems: 'center' }}>
                <span className="chip" style={{ minWidth: 0 }}>{metNameById(mid)}</span>
                <input className="input" type="number" min={0} step={0.1} value={vals.yield}
                  onChange={(e) => {
                    const y = Math.max(0, Number(e.target.value));
                    setFeedingTerms((prev) => ({ ...prev, [ftId]: { ...ft, inputs: { ...ft.inputs, [mid]: { ...vals, yield: y } } } }));
                  }} style={{ width: 90 }} />
                <span className="unit">yield</span>
                <input className="input" type="number" min={0.0001} step={0.1} value={vals.monodK}
                  onChange={(e) => {
                    const k = Math.max(0.0001, Number(e.target.value));
                    setFeedingTerms((prev) => ({ ...prev, [ftId]: { ...ft, inputs: { ...ft.inputs, [mid]: { ...vals, monodK: k } } } }));
                  }} style={{ width: 90 }} />
                <span className="unit">Monod K</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="form-row">
        <label className="label">Outputs (produced)</label>
        {Object.keys(ft.outputs).length === 0 ? (
          <div className="hint">Connect feeding term → metabolite to add outputs.</div>
        ) : (
          <div style={{ display: 'grid', gap: 6 }}>
            {Object.entries(ft.outputs).map(([mid, vals]) => (
              <div key={mid} className="inline" style={{ gap: 8, alignItems: 'center' }}>
                <span className="chip" style={{ minWidth: 0 }}>{metNameById(mid)}</span>
                <input className="input" type="number" min={0} step={0.1} value={vals.yield}
                  onChange={(e) => {
                    const y = Math.max(0, Number(e.target.value));
                    setFeedingTerms((prev) => ({ ...prev, [ftId]: { ...ft, outputs: { ...ft.outputs, [mid]: { yield: y } } } }));
                  }} style={{ width: 90 }} />
                <span className="unit">yield</span>
                <span className="muted" style={{ marginLeft: 6 }}>(Monod K = 0)</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}


