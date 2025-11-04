import React from "react";
import type { RFNode } from "../../../components/BuildCanvas";

export function SubpopulationInspector({
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
      <h2>Subpopulation</h2>
      <div className="form-row">
        <label className="label">Name</label>
        <input
          className="input"
          value={(selectedRf.data as any).name || 'subpop'}
          onChange={(e) => {
            const name = e.target.value;
            setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), name } } : n));
          }}
        />
      </div>
      <div className="form-row">
        <label className="label">Initial Count</label>
        <div className="inline">
          <input
            type="number"
            min={0}
            step={0.001}
            className="input"
            value={(selectedRf.data as any).count ?? 0.003}
            onChange={(e) => {
              const count = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), count } } : n));
            }}
          />
          <span className="unit">cells/mL</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">μmax (max growth rate)</label>
        <div className="inline">
          <input
            type="number"
            min={0}
            step={0.01}
            className="input"
            value={(selectedRf.data as any).mumax ?? 0.5}
            onChange={(e) => {
              const mumax = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), mumax } } : n));
            }}
          />
          <span className="unit">h⁻¹</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">Optimal pH</label>
        <div className="inline">
          <input
            type="number"
            min={0}
            max={14}
            step={0.1}
            className="input"
            value={(selectedRf.data as any).pHopt ?? 7.0}
            onChange={(e) => {
              const pHopt = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), pHopt } } : n));
            }}
          />
          <span className="unit">pH</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">pH Sensitivity (left)</label>
        <div className="inline">
          <input
            type="number"
            min={0.1}
            step={0.1}
            className="input"
            value={(selectedRf.data as any).pH_sensitivity_left ?? 2.0}
            onChange={(e) => {
              const pH_sensitivity_left = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), pH_sensitivity_left } } : n));
            }}
          />
          <span className="unit">pH units</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">pH Sensitivity (right)</label>
        <div className="inline">
          <input
            type="number"
            min={0.1}
            step={0.1}
            className="input"
            value={(selectedRf.data as any).pH_sensitivity_right ?? 2.0}
            onChange={(e) => {
              const pH_sensitivity_right = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), pH_sensitivity_right } } : n));
            }}
          />
          <span className="unit">pH units</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">Optimal Temperature</label>
        <div className="inline">
          <input
            type="number"
            min={0}
            max={100}
            step={0.5}
            className="input"
            value={(selectedRf.data as any).Topt ?? 37.0}
            onChange={(e) => {
              const Topt = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), Topt } } : n));
            }}
          />
          <span className="unit">°C</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">Temp Sensitivity (left)</label>
        <div className="inline">
          <input
            type="number"
            min={0.1}
            step={0.1}
            className="input"
            value={(selectedRf.data as any).tempSensitivity_left ?? 5.0}
            onChange={(e) => {
              const tempSensitivity_left = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), tempSensitivity_left } } : n));
            }}
          />
          <span className="unit">°C</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">Temp Sensitivity (right)</label>
        <div className="inline">
          <input
            type="number"
            min={0.1}
            step={0.1}
            className="input"
            value={(selectedRf.data as any).tempSensitivity_right ?? 2.0}
            onChange={(e) => {
              const tempSensitivity_right = Number(e.target.value);
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), tempSensitivity_right } } : n));
            }}
          />
          <span className="unit">°C</span>
        </div>
      </div>
      <div className="form-row">
        <label className="label">State</label>
        <select
          className="input"
          value={(selectedRf.data as any).state || 'active'}
          onChange={(e) => {
            const state = e.target.value;
            setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), state } } : n));
          }}
        >
          <option value="active">Active</option>
          <option value="dormant">Dormant</option>
          <option value="inactive">Inactive</option>
          <option value="dead">Dead</option>
        </select>
      </div>
      <div className="form-row">
        <label className="label">Color</label>
        <div className="inline grow">
          <input
            type="color"
            className="input"
            style={{ width: 52, height: 34, padding: 4 }}
            value={(selectedRf.data as any).color || '#aaaaaa'}
            onChange={(e) => {
              const color = e.target.value;
              setRfNodes((prev) => prev.map((n) => n.id === selectedRf.id ? { ...n, data: { ...(n.data as any), color } } : n));
            }}
          />
          <input
            className="input"
            value={(selectedRf.data as any).color || '#aaaaaa'}
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


