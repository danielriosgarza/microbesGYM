import React from "react";

export type SavedPulse = {
  id: string;
  name: string;
  t_start: number;
  t_end: number;
  n_steps: number;
};

export type DraftPulse = {
  id: string;
  name: string;
  t_start: number;
  t_end: number;
  n_steps: number;
};

type Kind = "saved" | "draft";

export interface PulseTimelineProps {
  saved: SavedPulse[];
  drafts: DraftPulse[];
  onSelect: (id: string | null, kind: Kind) => void;
  onDraftChange: (id: string, patch: Partial<DraftPulse>) => void;
  onAddDraft?: () => void;
  pxPerHour?: number;
  setPxPerHour?: (px: number) => void;
  snap?: number; // hours, default 0.25
  minDuration?: number; // hours, default 0.25
}

const DEFAULT_PX_PER_HOUR = 60;

export const PulseTimeline: React.FC<PulseTimelineProps> = ({
  saved,
  drafts,
  onSelect,
  onDraftChange,
  onAddDraft,
  pxPerHour = DEFAULT_PX_PER_HOUR,
  setPxPerHour,
  snap = 0.25,
  minDuration = 0.25,
}) => {
  const scrollerRef = React.useRef<HTMLDivElement | null>(null);
  const surfaceRef = React.useRef<HTMLDivElement | null>(null);
  const [drag, setDrag] = React.useState<
    | null
    | {
        type: "move" | "resize-left" | "resize-right";
        id: string;
        startX: number;
        startScroll: number;
        t0: number;
        t1: number;
      }
  >(null);

  const all = React.useMemo(
    () => [
      ...saved.map((p) => ({ ...p, kind: "saved" as Kind })),
      ...drafts.map((p) => ({ ...p, kind: "draft" as Kind })),
    ],
    [saved, drafts]
  );

  const contentHours = React.useMemo(() => {
    const end = Math.max(0, ...all.map((p) => p.t_end));
    return Math.max(8, Math.ceil(end + 1));
  }, [all]);

  const toPx = (h: number) => 24 + h * pxPerHour;
  const fromPx = (px: number) => (px - 24) / pxPerHour;
  const snapH = (h: number) => Math.round(h / snap) * snap;

  const nonOverlapCandidateOk = (id: string, t0: number, t1: number) => {
    for (const p of all) {
      if (p.id === id) continue;
      // Overlap if intervals intersect
      if (!(t1 <= p.t_start || t0 >= p.t_end)) {
        // block only if the other is saved or a different draft
        return false;
      }
    }
    return true;
  };

  const onMouseMove = React.useCallback(
    (e: MouseEvent) => {
      if (!drag) return;
      const sc = scrollerRef.current ? scrollerRef.current.scrollLeft : 0;
      const dx = e.clientX + sc - (drag.startX + drag.startScroll);
      const dt = dx / pxPerHour;
      let t0 = drag.t0;
      let t1 = drag.t1;
      if (drag.type === "move") {
        t0 = snapH(drag.t0 + dt);
        t1 = snapH(drag.t1 + dt);
      } else if (drag.type === "resize-left") {
        t0 = snapH(drag.t0 + dt);
        if (t1 - t0 < minDuration) t0 = t1 - minDuration;
      } else if (drag.type === "resize-right") {
        t1 = snapH(drag.t1 + dt);
        if (t1 - t0 < minDuration) t1 = t0 + minDuration;
      }
      // enforce non-overlap
      if (!nonOverlapCandidateOk(drag.id, Math.max(0, t0), Math.max(0, t1))) return;
      // draft only
      onDraftChange(drag.id, { t_start: Math.max(0, t0), t_end: Math.max(0, t1) });
    },
    [drag, minDuration, onDraftChange, pxPerHour]
  );

  const onMouseUp = React.useCallback(() => setDrag(null), []);

  React.useEffect(() => {
    if (!drag) return;
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, [drag, onMouseMove, onMouseUp]);

  return (
    <div
      ref={scrollerRef}
      style={{
        position: "relative",
        overflowX: "auto",
        overflowY: "hidden",
        border: "1px solid var(--border)",
        borderRadius: 10,
        background: "color-mix(in oklab, var(--panel), white 10%)",
        height: 180,
      }}
    >
      <div
        ref={surfaceRef}
        style={{ position: "relative", height: "100%", width: toPx(contentHours) + 48 }}
        onClick={(e) => {
          if (e.target === surfaceRef.current) onSelect(null, "draft");
        }}
      >
        {/* Axis ticks (1h) */}
        <div style={{ position: "absolute", top: 4, left: 0, right: 0, height: 16 }}>
          {Array.from({ length: Math.ceil(contentHours) + 1 }, (_, i) => (
            <div key={i} style={{ position: "absolute", left: toPx(i), top: 0, fontSize: 10 }}>
              {i}h
            </div>
          ))}
        </div>

        {/* Grid (0.5h) */}
        <div style={{ position: "absolute", top: 24, left: 0, right: 0, bottom: 0 }}>
          {Array.from({ length: Math.ceil(contentHours / 0.5) + 1 }, (_, i) => (
            <div
              key={i}
              style={{
                position: "absolute",
                left: toPx(i * 0.5),
                top: 0,
                bottom: 0,
                width: 1,
                background: i % 2 === 0 ? "#1f2937" : "#374151",
                opacity: 0.2,
              }}
            />
          ))}
        </div>

        {/* Saved pulses */}
        {saved.map((p) => {
          const left = toPx(p.t_start);
          const width = (p.t_end - p.t_start) * pxPerHour;
          const label = p.name && p.name.trim() ? p.name : "pulse";
          return (
            <div
              key={p.id}
              className="pulse-saved"
              title={`${label}: ${p.t_start}-${p.t_end}h`}
              style={{
                position: "absolute",
                left,
                top: 48,
                width,
                height: 64,
                borderRadius: 8,
                background: "#2563eb",
                border: "1px solid #1d4ed8",
                color: "white",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 12,
                fontWeight: 700,
                userSelect: "none",
              }}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(p.id, "saved");
              }}
            >
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", lineHeight: 1.1 }}>
                <span
                  aria-label="saved"
                  title="Saved"
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 999,
                    background: "#22d3ee",
                    boxShadow: "0 0 0 1px rgba(0,0,0,0.3)",
                    marginBottom: 4,
                    pointerEvents: "none",
                  }}
                />
                <div style={{ fontWeight: 700 }}>{label}</div>
                <div style={{ opacity: 0.85 }}>{`${p.t_start}-${p.t_end}h`}</div>
              </div>
            </div>
          );
        })}

        {/* Draft pulses (draggable) */}
        {drafts.map((p) => {
          const left = toPx(p.t_start);
          const width = (p.t_end - p.t_start) * pxPerHour;
          const label = p.name && p.name.trim() ? p.name : "pulse";
          return (
            <div
              key={p.id}
              className="pulse-draft"
              title={`${label}: ${p.t_start}-${p.t_end}h`}
              style={{
                position: "absolute",
                left,
                top: 48,
                width,
                height: 64,
                borderRadius: 8,
                background: "#10b981",
                border: "1px solid #059669",
                color: "#01210f",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 12,
                fontWeight: 700,
                userSelect: "none",
                cursor: "grab",
              }}
              onMouseDown={(e) => {
                onSelect(p.id, "draft");
                setDrag({
                  type: "move",
                  id: p.id,
                  startX: e.clientX,
                  startScroll: scrollerRef.current?.scrollLeft || 0,
                  t0: p.t_start,
                  t1: p.t_end,
                });
              }}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(p.id, "draft");
              }}
            >
              <div
                style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 8, cursor: "ew-resize" }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onSelect(p.id, "draft");
                  setDrag({
                    type: "resize-left",
                    id: p.id,
                    startX: e.clientX,
                    startScroll: scrollerRef.current?.scrollLeft || 0,
                    t0: p.t_start,
                    t1: p.t_end,
                  });
                }}
              />
              <div
                style={{ position: "absolute", right: 0, top: 0, bottom: 0, width: 8, cursor: "ew-resize" }}
                onMouseDown={(e) => {
                  e.stopPropagation();
                  onSelect(p.id, "draft");
                  setDrag({
                    type: "resize-right",
                    id: p.id,
                    startX: e.clientX,
                    startScroll: scrollerRef.current?.scrollLeft || 0,
                    t0: p.t_start,
                    t1: p.t_end,
                  });
                }}
              />
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", lineHeight: 1.1 }}>
                <span
                  aria-label="unsaved"
                  title="Draft (unsaved)"
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: 999,
                    background: "#f87171",
                    boxShadow: "0 0 0 1px rgba(0,0,0,0.3)",
                    marginBottom: 4,
                    pointerEvents: "none",
                  }}
                />
                <div style={{ fontWeight: 700 }}>{label}</div>
                <div style={{ opacity: 0.85 }}>{`${p.t_start}-${p.t_end}h`}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PulseTimeline;
