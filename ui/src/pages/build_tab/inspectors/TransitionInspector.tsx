import React from "react";
import type { RFNode } from "../../../components/BuildCanvas";

export function TransitionInspector({
  selectedRf,
  selectedId,
  setSelectedId,
  setRfNodes,
}: {
  selectedRf: RFNode;
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  setRfNodes: React.Dispatch<React.SetStateAction<RFNode[]>>;
}) {
  return (
    <div className="form">
      <h2>Transition</h2>
      <div className="form-row">
        <label className="label">Rate</label>
        <div className="inline">
          <input
            type="number"
            min={0}
            step={0.01}
            className="input"
            value={(selectedRf.data as any).rate || 0.1}
            onChange={(e) => {
              const rate = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), rate } } : n));
            }}
          />
          <span className="unit">h⁻¹</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">Condition</label>
        <textarea
          className="input"
          value={(selectedRf.data as any).condition || ''}
          onChange={(e) => {
            const condition = e.target.value;
            setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), condition } } : n));
          }}
          rows={3}
          placeholder="e.g., pH < 6.5, glucose > 10 mM"
        />
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


