import React from "react";
import { Tabs } from "./components/Tabs";
import { Header } from "./components/Header";
import { Build } from "./pages/Build";
import { Simulation } from "./pages/Simulation";
import { Plots } from "./pages/Plots";
import { Training } from "./pages/Training";
import { Policies } from "./pages/Policies";

type TabId = "build" | "simulation" | "plots" | "training" | "policies";

const TABS: { id: TabId; label: string }[] = [
  { id: "build", label: "Build" },
  { id: "simulation", label: "Simulation" },
  { id: "plots", label: "Plots" },
  { id: "training", label: "Training" },
  { id: "policies", label: "Policies" }
];

function usePersistentState<T>(key: string, initial: T) {
  const [state, setState] = React.useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  React.useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      /* ignore */
    }
  }, [key, state]);
  return [state, setState] as const;
}

export default function App() {
  const [active, setActive] = usePersistentState<TabId>("mg-active-tab", "build");

  return (
    <div className="app-root">
      <Header />
      <nav className="tabs container" role="tablist" aria-label="Sections">
        <Tabs
          tabs={TABS}
          activeId={active}
          onChange={(id) => setActive(id as TabId)}
        />
      </nav>
      <main className="container">
        <section
          className={`panel ${active === "build" ? "" : "hidden"}`}
          role="tabpanel"
          aria-labelledby="tab-build"
        >
          <Build />
        </section>
        <section
          className={`panel ${active === "simulation" ? "" : "hidden"}`}
          role="tabpanel"
          aria-labelledby="tab-simulation"
        >
          <Simulation />
        </section>
        <section
          className={`panel ${active === "plots" ? "" : "hidden"}`}
          role="tabpanel"
          aria-labelledby="tab-plots"
        >
          <Plots />
        </section>
        <section
          className={`panel ${active === "training" ? "" : "hidden"}`}
          role="tabpanel"
          aria-labelledby="tab-training"
        >
          <Training />
        </section>
        <section
          className={`panel ${active === "policies" ? "" : "hidden"}`}
          role="tabpanel"
          aria-labelledby="tab-policies"
        >
          <Policies />
        </section>
      </main>
      <footer className="site-footer">
        <div className="container row space-between center">
          <span className="muted">v0.1 — UI prototype</span>
          <span className="muted">No data leaves your machine</span>
        </div>
      </footer>
    </div>
  );
}
