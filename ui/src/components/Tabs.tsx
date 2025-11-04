import React from "react";

type Tab = { id: string; label: string };

export function Tabs({
  tabs,
  activeId,
  onChange
}: {
  tabs: Tab[];
  activeId: string;
  onChange: (id: string) => void;
}) {
  const listRef = React.useRef<HTMLDivElement>(null);
  const ids = tabs.map((t) => t.id);

  const onKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>) => {
    const current = (e.currentTarget as HTMLButtonElement).dataset.tab!;
    const i = ids.indexOf(current);
    if (i < 0) return;
    let nextIndex = i;
    if (e.key === "ArrowRight") nextIndex = (i + 1) % ids.length;
    else if (e.key === "ArrowLeft") nextIndex = (i - 1 + ids.length) % ids.length;
    else if (e.key === "Home") nextIndex = 0;
    else if (e.key === "End") nextIndex = ids.length - 1;
    else return;
    e.preventDefault();
    const nextId = ids[nextIndex];
    const el = listRef.current?.querySelector<HTMLButtonElement>(`button[data-tab="${nextId}"]`);
    el?.focus();
    onChange(nextId);
  };

  return (
    <div className="tablist" ref={listRef}>
      {tabs.map((t) => {
        const selected = t.id === activeId;
        return (
          <button
            key={t.id}
            id={`tab-${t.id}`}
            className={`tab ${selected ? "active" : ""}`}
            data-tab={t.id}
            role="tab"
            aria-selected={selected}
            aria-controls={`panel-${t.id}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(t.id)}
            onKeyDown={onKeyDown}
            type="button"
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

