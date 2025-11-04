// Concurrent dev runner: bootstraps Python venv, starts API and Vite, forwards logs, cleans up on exit
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const FRONTEND_CWD = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(FRONTEND_CWD, "..");

function run(name, command, args, opts = {}) {
  const child = spawn(command, args, {
    cwd: opts.cwd || FRONTEND_CWD,
    stdio: "inherit",
    shell: true,
    env: { ...process.env, UV_LINK_MODE: "copy", ...(opts.env || {}) },
  });
  child.on("exit", (code, signal) => {
    console.log(`[${name}] exited with`, signal || code);
  });
  return child;
}

function runAsync(name, command, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const child = run(name, command, args, opts);
    child.on("exit", (code) => {
      if (code === 0) return resolve();
      reject(new Error(`${name} exited with code ${code}`));
    });
    child.on("error", reject);
  });
}

// Minimal runner: rely on uvx to resolve Python tool deps (uvicorn, fastapi)

const procs = [];

function stopAll() {
  for (const p of procs) {
    if (!p || p.killed) continue;
    try {
      if (process.platform === "win32") {
        // Best-effort kill tree on Windows
        spawn("taskkill", ["/pid", String(p.pid), "/T", "/F"], { shell: true });
      } else {
        p.kill("SIGTERM");
      }
    } catch {}
  }
}

process.on("SIGINT", () => {
  console.log("\nStopping dev processes...");
  stopAll();
  process.exit(0);
});
process.on("SIGTERM", () => {
  stopAll();
  process.exit(0);
});
process.on("exit", () => {
  stopAll();
});

(async () => {
  try {
    console.log("Starting API (uvicorn) and Vite dev server...\n");

    // Start FastAPI via uvx from repo root, pointing at src
    procs.push(
      run(
        "api",
        "uvx",
        [
          // Ensure required libs are available in the runner env
          "--with",
          "fastapi,pydantic,numpy,scipy,plotly,pandas",
          "uvicorn",
          "--app-dir",
          path.join("src"),
          "--reload-dir",
          path.join("src"),
          "mg_api.main:app",
          "--reload",
          "--port",
          "8000",
        ],
        {
          cwd: REPO_ROOT,
          env: {},
        }
      )
    );

    // Start Vite dev server
    procs.push(run("web", "npm", ["run", "web"], { cwd: FRONTEND_CWD }));
  } catch (err) {
    console.error("Failed to start dev environment:\n", err?.stack || err);
    process.exit(1);
  }
})();
