import React from "react";
import type { MetaboliteIn } from "../../../lib/metabolites";

export function MetaboliteInspector({
  selected,
  errors,
  saving,
  savedAt,
  updateSelected,
  saveSelected,
  deleteSelected,
}: {
  selected: { localId: string; saved: boolean; lastSaved?: any; data: MetaboliteIn } | null;
  errors: Record<string, string>;
  saving: boolean;
  savedAt: number | null;
  updateSelected: (patch: Partial<MetaboliteIn>) => void;
  saveSelected: () => Promise<void> | void;
  deleteSelected: () => Promise<void> | void;
}) {
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

  return (
    <>
      <h2>model inspector</h2>
      {!selected ? (
        <p className="muted">Select a node to edit its properties.</p>
      ) : (
        <div className="form">
          <div className="form-row">
            <label className="label" htmlFor="name">Name</label>
            <input
              id="name"
              className={`input${errors.name ? " invalid" : ""}`}
              value={selected.data.name}
              onChange={(e) => updateSelected({ name: e.target.value })}
            />
            {errors.name ? <div className="hint error">{errors.name}</div> : <div className="hint">Unique within model</div>}
          </div>

          <div className="form-row">
            <label className="label" htmlFor="conc">Concentration</label>
            <div className="inline">
              <input
                id="conc"
                type="number"
                min={0}
                step={0.1}
                className={`input${errors.concentration ? " invalid" : ""}`}
                value={selected.data.concentration}
                onChange={(e) => updateSelected({ concentration: Number(e.target.value) })}
              />
              <span className="unit">mM</span>
            </div>
            {errors.concentration ? <div className="hint error">{errors.concentration}</div> : <div className="hint">Non-negative</div>}
          </div>

          <div className="form-row">
            <label className="label">Color</label>
            <div className="inline grow">
              <input
                type="color"
                className="input"
                style={{ width: 52, height: 34, padding: 4 }}
                value={selected.data.color}
                onChange={(e) => updateSelected({ color: e.target.value })}
              />
              <input
                className="input"
                value={selected.data.color}
                onChange={(e) => updateSelected({ color: e.target.value })}
                style={{ minWidth: 0 }}
              />
            </div>
            <div className="swatches">
              {["#22d3ee","#a78bfa","#34d399","#fbbf24","#f87171"].map((c) => (
                <button key={c} className="swatch" style={{ background: c }} onClick={() => updateSelected({ color: c })} aria-label={`Set color ${c}`} />
              ))}
            </div>
          </div>

          <div className="form-row">
            <label className="label">Formula</label>
            <FormulaEditor
              value={selected.data.formula}
              onChange={(f) => updateSelected({ formula: f })}
            />
            {errors.formula ? <div className="hint error">{errors.formula}</div> : <div className="hint">Integer element counts</div>}
          </div>

          <div className="form-row">
            <label className="label" htmlFor="desc">Description</label>
            <textarea
              id="desc"
              className="input"
              value={selected.data.description}
              onChange={(e) => updateSelected({ description: e.target.value })}
              rows={3}
            />
          </div>

          <div className="row" style={{ gap: 8, alignItems: "center" }}>
            <button className="btn primary" onClick={saveSelected} type="button" disabled={true} style={{ display: 'none' }}>
              {saving ? "Saving…" : selected.saved ? "Save again" : "Save"}
            </button>
            <button className="btn" onClick={deleteSelected} type="button" disabled={saving}>
              Delete
            </button>
            {(() => {
              const ls = selected.lastSaved as any;
              const d = selected.data as any;
              const dirty = !ls || ls.name !== d.name || ls.color !== d.color || ls.concentration !== d.concentration ||
                ["C","H","O","N","S","P"].some((k) => (ls?.formula as any)?.[k] !== (d.formula as any)[k]) || (ls?.description || "") !== (d.description || "");
              if (dirty) return <span className="hint">Unsaved changes</span>;
              if (savedAt) return <span className="hint">Saved</span>;
              return null;
            })()}
          </div>
        </div>
      )}
    </>
  );
}


