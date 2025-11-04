import React from "react";
import { ApiStatus } from "../components/ApiStatus";

export function Training() {
  return (
    <div className="grid" id="panel-training">
      <div className="card" style={{ gridColumn: "span 12" }}>
        <ApiStatus />
      </div>
      <div className="card">
        <h2>Training</h2>
        <p className="muted">Configure RL algorithm and reward shaping. (Stub)</p>
        <div className="placeholder">
          <div className="skeleton title"></div>
          <div className="skeleton line"></div>
          <div className="skeleton line"></div>
        </div>
      </div>
    </div>
  );
}
