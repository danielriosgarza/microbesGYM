import React from "react";

export function Header() {
  return (
    <header className="site-header">
      <div className="container row space-between center">
        <div className="brand row center">
          <div className="logo" aria-hidden="true">μGym</div>
          <div className="brand-text">
            <h1>Microbes Gym</h1>
            <p className="muted">Train and evaluate microbiome controllers</p>
          </div>
        </div>
        <div className="header-actions row center"></div>
      </div>
    </header>
  );
}

