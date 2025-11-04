import React from "react";
import { ApiStatus } from "../components/ApiStatus";

export function Policies() {
  return (
    <div className="grid" id="panel-policies">
      <div className="card" style={{ gridColumn: "span 12" }}>
        <ApiStatus />
      </div>
      <div className="card">
        <h2>Policies</h2>
        <p className="muted">Inspect, compare, and evaluate learned policies. (Stub)</p>
        <div className="placeholder">
          <div className="skeleton title"></div>
          <div className="skeleton line"></div>
          <div className="skeleton line"></div>
        </div>
      </div>
    </div>
  );
}
