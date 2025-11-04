// Lightweight loader for Plotly from CDN, avoids adding npm dep.
export async function ensurePlotly(): Promise<any> {
  if ((window as any).Plotly) return (window as any).Plotly;
  const src = "https://cdn.plot.ly/plotly-2.30.0.min.js";
  await new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load Plotly.js"));
    document.head.appendChild(s);
  });
  return (window as any).Plotly;
}

