import React from "react";
import { getHealth } from "../lib/api";

export function ApiStatus() {
  const [online, setOnline] = React.useState<boolean | null>(null);
  const [loading, setLoading] = React.useState(false);

  const check = React.useCallback(async () => {
    setLoading(true);
    try {
      const ctl = new AbortController();
      const res = await getHealth(ctl.signal);
      setOnline(!!res && res.status === "ok");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, [check]);

  const color = online === null ? "#9fb0c3" : online ? "#22d3ee" : "#f87171";
  const label = online === null ? "Checking" : online ? "Online" : "Offline";

  return (
    <div
      className="api-status muted"
      title="API health"
      style={{ display: "flex", alignItems: "center" }}
    >
      <span
        aria-hidden
        style={{
          display: "inline-block",
          width: 8,
          height: 8,
          borderRadius: 999,
          background: color,
          marginRight: 8,
          boxShadow: online ? "0 0 8px rgba(34,211,238,0.35)" : undefined
        }}
      />
      <span>API: {label}</span>
      <button
        type="button"
        onClick={check}
        disabled={loading}
        className="btn"
        style={{ marginLeft: 10, padding: "6px 10px", fontSize: 12 }}
      >
        {loading ? "Checking..." : "Retry"}
      </button>
    </div>
  );
}

