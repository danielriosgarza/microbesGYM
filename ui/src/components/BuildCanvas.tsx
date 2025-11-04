import React from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Edge,
  Node,
  NodeProps,
  NodeTypes,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  addEdge,
  useEdgesState,
  applyNodeChanges,
  Handle,
  Position,
  useReactFlow,
} from "reactflow";
import "reactflow/dist/style.css";

export type MetaboliteNodeData = {
  name: string;
  concentration: number;
  color: string;
  saved?: boolean;
};

export type PHNodeData = {
  baseValue: number;
};

export type FeedingTermNodeData = { name: string; inputs?: number; outputs?: number };
export type SubpopulationNodeData = { name: string; mumax: number; state: "active" | "inactive" | "dead"; color?: string };
export type TransitionNodeData = { rate: number; condition: string };
export type BacteriaNodeData = { species: string; color: string };

export type RFNode = Node<
  MetaboliteNodeData | PHNodeData | FeedingTermNodeData | SubpopulationNodeData | TransitionNodeData | BacteriaNodeData
>;

function MetaboliteNode({ data, selected }: NodeProps<MetaboliteNodeData>) {
  const size = 64; // smaller default size
  return (
    <div style={{ display: "grid", placeItems: "center" }}>
      {/* Top pill tag */}
      <div
        style={{
          marginBottom: 6,
          padding: "2px 8px",
          borderRadius: 999,
          fontSize: 10,
          background: "linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%)",
          color: "#02050a",
          border: "1px solid color-mix(in oklab, var(--primary), var(--accent) 35%)",
          boxShadow: "var(--glow)",
        }}
      >
        molecule
      </div>

      {/* Circle with name inside */}
      <div
        style={{
          width: size,
          height: size,
          borderRadius: "50%",
          border: selected ? `2px solid var(--primary)` : `2px solid ${data.color}`,
          background: "color-mix(in oklab, var(--panel), #000 10%)",
          boxShadow: data.saved ? "var(--shadow)" : "0 0 0 3px rgba(250,204,21,0.25)",
          display: "grid",
          placeItems: "center",
          textAlign: "center",
          padding: 4,
        }}
        title={`${data.name} (${data.concentration} mM)`}
      >
        <div style={{ display: "grid", placeItems: "center", gap: 2 }}>
          <div style={{ fontSize: 11, lineHeight: 1.1 }}>{data.name}</div>
          <div className="muted" style={{ fontSize: 9 }}>{data.concentration} mM</div>
        </div>
      </div>


      {/* Handles (enabled), validation will block metabolite↔metabolite */}
      <Handle type="target" position={Position.Left} style={{ background: "var(--accent)", width: 8, height: 8 }} />
      <Handle type="source" position={Position.Right} style={{ background: "var(--primary)", width: 8, height: 8 }} />
      </div>
  );
}

function PHNode({ data, selected }: NodeProps<PHNodeData>) {
  // Approximately 30% of the older 72px size
  const size = 22;
  return (
    <div style={{ display: "grid", placeItems: "center" }} title={`Base pH: ${(data as PHNodeData).baseValue}`}>
      {/* Label */}
      <div
        style={{
          marginBottom: 4,
          padding: "1px 6px",
          borderRadius: 999,
          fontSize: 9,
          background: "linear-gradient(135deg, var(--accent) 0%, var(--primary) 100%)",
          color: "#02050a",
          border: "1px solid color-mix(in oklab, var(--accent), var(--primary) 35%)",
          boxShadow: "var(--glow)",
        }}
      >
        pH
      </div>

      {/* Diamond shape via rotation */}
      <div
        style={{
          width: size,
          height: size,
          transform: "rotate(45deg)",
          border: selected ? `2px solid var(--accent)` : `2px solid var(--border)`,
          background: "color-mix(in oklab, var(--panel), #000 10%)",
          boxShadow: "var(--shadow)",
        }}
      />

      {/* Single target handle to receive metabolite connections */}
      <Handle type="target" position={Position.Left} style={{ background: "var(--accent)", width: 8, height: 8 }} />
    </div>
  );
}

function FeedingTermNode({ data, selected }: NodeProps<FeedingTermNodeData>) {
  const size = 64;
  return (
    <div style={{ display: "grid", placeItems: "center" }} title={`Feeding term`}>
      <div style={{ marginBottom: 6, padding: "2px 8px", borderRadius: 999, fontSize: 10, background: "linear-gradient(135deg, #10b981 0%, #22d3ee 100%)", color: "#02050a", border: "1px solid color-mix(in oklab, #10b981, #22d3ee 35%)", boxShadow: "var(--glow)" }}>feeding</div>
      <div style={{ width: size, height: size, borderRadius: 12, border: selected ? `2px solid #10b981` : `2px solid var(--border)`, background: "color-mix(in oklab, var(--panel), #000 10%)", boxShadow: "var(--shadow)", display: "grid", placeItems: "center", padding: 6, textAlign: "center", lineHeight: 1.1, fontSize: 11 }}>
        <div style={{ display: "grid", gap: 2 }}>
          <div style={{ fontWeight: 700 }}>{data.name || "feeding"}</div>
          <div className="muted" style={{ fontSize: 9 }}>{`${data.inputs ?? 0} in · ${data.outputs ?? 0} out`}</div>
        </div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: "#22d3ee", width: 8, height: 8 }} />
      <Handle type="source" position={Position.Right} style={{ background: "#10b981", width: 8, height: 8 }} />
    </div>
  );
}

function SubpopulationNode({ data, selected }: NodeProps<SubpopulationNodeData>) {
  const size = 64;
  const badgeColor = data.state === "active" ? "#22d3ee" : data.state === "inactive" ? "#fbbf24" : "#f87171";
  return (
    <div style={{ display: "grid", placeItems: "center" }} title={`Subpopulation: ${data.name}`}>
      <div style={{ marginBottom: 6, padding: "2px 8px", borderRadius: 999, fontSize: 10, background: "linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%)", color: "#02050a", border: "1px solid color-mix(in oklab, var(--primary), var(--accent) 35%)", boxShadow: "var(--glow)" }}>subpopulation</div>
      <div style={{ width: size, height: size, borderRadius: 12, border: selected ? `2px solid var(--primary)` : `2px solid var(--border)`, background: "color-mix(in oklab, var(--panel), #000 10%)", boxShadow: "var(--shadow)", display: "grid", placeItems: "center", padding: 6, textAlign: "center", lineHeight: 1.1, fontSize: 11 }}>
        <div style={{ display: "grid", gap: 2 }}>
          <div style={{ fontWeight: 700 }}>{data.name || "subpop"}</div>
          <div className="muted" style={{ fontSize: 9 }}>μmax {data.mumax}</div>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
            <span aria-hidden style={{ width: 8, height: 8, borderRadius: 999, background: badgeColor }} />
            <span className="muted" style={{ fontSize: 9 }}>{data.state}</span>
          </div>
        </div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: "var(--accent)", width: 8, height: 8 }} />
      <Handle type="source" position={Position.Right} style={{ background: "var(--primary)", width: 8, height: 8 }} />
    </div>
  );
}

function TransitionNode({ data, selected }: NodeProps<TransitionNodeData>) {
  const size = 48;
  return (
    <div style={{ display: "grid", placeItems: "center" }} title={`Transition: rate ${data.rate}`}>
      <div style={{ marginBottom: 4, padding: "1px 6px", borderRadius: 999, fontSize: 9, background: "linear-gradient(135deg, #f59e0b 0%, #f97316 100%)", color: "#02050a", border: "1px solid color-mix(in oklab, #f59e0b, #f97316 35%)", boxShadow: "var(--glow)" }}>transition</div>
      <div style={{ width: size, height: size, transform: "rotate(45deg)", border: selected ? `2px solid #f59e0b` : `2px solid var(--border)`, background: "color-mix(in oklab, var(--panel), #000 10%)", boxShadow: "var(--shadow)", display: "grid", placeItems: "center" }} />
      <div className="muted" style={{ fontSize: 9, marginTop: 4 }}>{data.rate} | {data.condition?.slice(0, 10) || ""}</div>
      <Handle type="target" position={Position.Left} style={{ background: "#f59e0b", width: 8, height: 8 }} />
      <Handle type="source" position={Position.Right} style={{ background: "#f59e0b", width: 8, height: 8 }} />
    </div>
  );
}

function BacteriaNode({ data, selected }: NodeProps<BacteriaNodeData>) {
  const size = 64;
  return (
    <div style={{ display: "grid", placeItems: "center" }} title={`Bacteria: ${data.species}`}>
      <div style={{ marginBottom: 6, padding: "2px 8px", borderRadius: 999, fontSize: 10, background: "linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%)", color: "#02050a", border: "1px solid color-mix(in oklab, #60a5fa, #a78bfa 35%)", boxShadow: "var(--glow)" }}>microbe</div>
      <div style={{ width: size, height: size, borderRadius: 12, border: selected ? `2px solid ${data.color}` : `2px solid var(--border)`, background: "color-mix(in oklab, var(--panel), #000 10%)", boxShadow: "var(--shadow)", display: "grid", placeItems: "center", padding: 6, textAlign: "center", lineHeight: 1.1, fontSize: 11 }}>
        <div style={{ display: "grid", gap: 2 }}>
          <div style={{ fontWeight: 700 }}>{data.species || "species"}</div>
        </div>
      </div>
      <Handle type="target" position={Position.Left} style={{ background: data.color || "#a78bfa", width: 8, height: 8 }} />
    </div>
  );
}

const nodeTypes: NodeTypes = {
  metaboliteNode: MetaboliteNode,
  phNode: PHNode,
  feedingTermNode: FeedingTermNode,
  subpopulationNode: SubpopulationNode,
  transitionNode: TransitionNode,
  bacteriaNode: BacteriaNode,
};

export function BuildCanvas({
  nodes,
  initialEdges,
  onNodesUpdate,
  onSelect,
  onEdgeCreate,
  onEdgeRemove,
  centerOnNodeId,
  backgroundColor,
  gridColor,
}: {
  nodes: RFNode[];
  initialEdges?: Array<{ id: string; source: string; target: string }>;
  onNodesUpdate?: (nodes: RFNode[]) => void;
  onSelect?: (id: string | null) => void;
  onEdgeCreate?: (edge: { source: string; target: string }) => void;
  onEdgeRemove?: (edge: { source: string; target: string }) => void;
  centerOnNodeId?: string | null;
  backgroundColor?: string;
  gridColor?: string;
}) {
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge[]>([]);
  // Sync incoming initial edges (e.g., when loading a preset)
  React.useEffect(() => {
    if (!initialEdges) return;
    setEdges(initialEdges as unknown as Edge[]);
  }, [initialEdges, setEdges]);
  const rf = useReactFlow();
  // Center on a given node id when prop changes
  React.useEffect(() => {
    if (!centerOnNodeId) return;
    // Try to read node from React Flow instance to avoid re-centering on every nodes change
    const n = (rf as any).getNode ? (rf as any).getNode(centerOnNodeId) : nodes.find((x) => x.id === centerOnNodeId);
    if (!n) return;
    try { rf.setCenter(n.position.x, n.position.y, { zoom: 1.0, duration: 300 }); } catch {}
    // Center only once per id change
  }, [centerOnNodeId, rf]);

  const onConnect: OnConnect = React.useCallback(
    (connection) => {
      setEdges((eds) => addEdge(connection, eds));
      if (connection.source && connection.target) {
        onEdgeCreate?.({ source: connection.source, target: connection.target });
      }
    },
    [setEdges, onEdgeCreate]
  );
  const isValidConnection = React.useCallback((conn: { source?: string | null; target?: string | null }) => {
    if (!conn.source || !conn.target) return false;
    if (conn.source === conn.target) return false;
    const src = nodes.find((n) => n.id === conn.source);
    const tgt = nodes.find((n) => n.id === conn.target);
    if (!src || !tgt) return false;
    // Block metabolite↔metabolite for now
    const isMetab = (n: RFNode) => n.type === "metaboliteNode";
    if (isMetab(src) && isMetab(tgt)) return false;
    // Allow anything else (future: feedingTerm, pH, etc.)
    return true;
  }, [nodes, edges]);

  const handleNodesChange: OnNodesChange = React.useCallback(
    (changes) => {
      const updated = applyNodeChanges(changes, nodes);
      onNodesUpdate?.(updated as RFNode[]);
    },
    [nodes, onNodesUpdate]
  );

  const handleEdgesChange: OnEdgesChange = React.useCallback(
    (changes) => {
      for (const ch of changes as any[]) {
        if (ch.type === "remove" && ch.id) {
          const edge = (edges as any[]).find((e) => e.id === ch.id);
          if (edge && edge.source && edge.target) {
            onEdgeRemove?.({ source: edge.source, target: edge.target });
          }
        }
      }
      onEdgesChange(changes);
    },
    [edges, onEdgesChange, onEdgeRemove]
  );

  // Override connection rules to enforce allowed pairs
  const isValidConnection2 = React.useCallback((conn: { source?: string | null; target?: string | null }) => {
    if (!conn.source || !conn.target) return false;
    if (conn.source === conn.target) return false;
    const src = nodes.find((n) => n.id === conn.source);
    const tgt = nodes.find((n) => n.id === conn.target);
    if (!src || !tgt) return false;
    const s = String(src.type);
    const t = String(tgt.type);
    if (s === "metaboliteNode" && t === "metaboliteNode") return false;
    if (s === "metaboliteNode" && t === "phNode") return true;
    if (s === "metaboliteNode" && t === "feedingTermNode") return true; // inputs
    if (s === "feedingTermNode" && t === "metaboliteNode") return true; // outputs
    if (s === "feedingTermNode" && t === "subpopulationNode") return true;
    if (s === "subpopulationNode" && t === "transitionNode") return true;
    if (s === "transitionNode" && t === "subpopulationNode") return true;
    if (s === "subpopulationNode" && t === "bacteriaNode") return true;
    return false;
  }, [nodes]);

  return (
    <div
      style={{ width: "100%", height: "clamp(520px, 80vh, 1100px)", background: backgroundColor || "transparent", borderRadius: 12, overflow: "hidden", border: "1px solid var(--border)" }}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Escape") onSelect?.(null);
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={onConnect}
        isValidConnection={isValidConnection2}
        nodeTypes={nodeTypes}
        snapToGrid
        snapGrid={[4, 4]}
        nodesDraggable={true}
        panOnDrag
        selectionOnDrag={false}
        panOnScroll
        zoomOnPinch
        zoomOnDoubleClick={false}
        onSelectionChange={(sel) => {
          const id = sel?.nodes && sel.nodes.length > 0 ? sel.nodes[0].id : null;
          onSelect?.(id);
        }}
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <Background variant="dots" gap={16} size={1} color={gridColor || "rgba(255,255,255,0.06)"} />
        <MiniMap pannable zoomable />
        <Controls position="bottom-right" />
      </ReactFlow>
    </div>
  );
}

// Center viewport on a node when requested
export function useCenterOnNode(rfNodes: RFNode[], centerOnNodeId?: string | null) {
  const rf = useReactFlow();
  React.useEffect(() => {
    if (!centerOnNodeId) return;
    const n = rfNodes.find((x) => x.id === centerOnNodeId);
    if (!n) return;
    const { x, y } = n.position;
    try {
      rf.setCenter(x, y, { zoom: 1.0, duration: 300 });
    } catch {}
  }, [centerOnNodeId, rfNodes, rf]);
}
