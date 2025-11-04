import React from "react";
import type { RFNode } from "../../../components/BuildCanvas";

export interface PHState {
  id: string;
  baseValue: number;
  weights: Record<string, number>;
  saved?: boolean;
}

export function PHInspector({
  ph,
  selectedId,
  setSelectedId,
  setPh,
  setRfNodes,
}: {
  ph: PHState;
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  setPh: React.Dispatch<React.SetStateAction<PHState | null>>;
  setRfNodes: React.Dispatch<React.SetStateAction<RFNode[]>>;
}) {
  return (
    <div className="form" style={{ marginBottom: 12 }}>
      <h2>pH</h2>
      <div className="form-row">
        <label className="label" htmlFor="ph-base">Base value</label>
        <div className="inline">
          <input
            id="ph-base"
            type="number"
            min={0}
            max={14}
            step={0.1}
            className="input"
            value={ph.baseValue}
            onChange={(e) => {
              const v = Math.max(0, Math.min(14, Number(e.target.value)));
              setPh((prev) => prev ? { ...prev, baseValue: v, saved: false } : prev);
              setRfNodes((prev) => prev.map((n) => n.id === ph.id ? { ...n, data: { ...(n.data as any), baseValue: v } } : n));
            }}
          />
          <span className="unit">pH</span>
        </div>
        <div className="hint">0 to 14</div>
      </div>

      <div className="form-row">
        <label className="label">Weights</label>
        {Object.keys(ph.weights).length === 0 ? (
          <div className="hint">Connect metabolites to the pH node to add weights.</div>
        ) : (
          <div style={{ display: 'grid', gap: 6 }}>
            {Object.entries(ph.weights).map(([name, w]) => (
              <div key={name} className="inline" style={{ gap: 6, alignItems: 'center' }}>
                <span className="chip" style={{ minWidth: 0 }}>{name}</span>
                <input
                  type="number"
                  className="input"
                  step={0.1}
                  value={w}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setPh((prev) => prev ? { ...prev, weights: { ...prev.weights, [name]: v }, saved: false } : prev);
                  }}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="row" style={{ gap: 8, alignItems: 'center' }}>
        <button
          className="btn"
          onClick={() => {
            if (!ph) return;
            setRfNodes((prev) => prev.filter((n) => n.id !== ph.id));
            setPh(null);
            if (selectedId === (ph && ph.id)) setSelectedId(null);
          }}
          type="button"
        >
          Delete pH node
        </button>
      </div>
    </div>
  );
}


