import katex from "katex";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { Backend, BackendEvent, b64F32, b64U32 } from "./api";
import { LossChart, ParetoChart, fmtLoss, HistPoint } from "./charts";
import { categorical, regionSlots } from "./palette";
import { Viewer, ColorMode } from "./viewer";

interface Hof { complexity: number; loss: number; score: number; equation: string; latex?: string | null }
interface Region {
  id: number;
  prim: string;
  n_points: number;
  area: number | null;
  frame: { kind: string; origin: number[]; e3: number[]; scale: number; var_names: string[] };
  status: string;
  hof: Hof[];
  selected: number | null;
  history: HistPoint[];
  iteration?: number;
  maxIter?: number;
  targetLoss?: number;
  nTrain?: number;
  guesses?: string[];
  dt?: number;
  latex?: string;
  world?: string;
  world_latex?: string;
  feature?: { kind: string; diameter: number; length: number } | null;
  params?: { tube_r?: number } & Record<string, unknown>;
  rms?: number;
  rms_rel?: number;
  max_err?: number;
  classification?: { type: string; [k: string]: any };
}

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number | null | undefined, d = 3) =>
  v == null || !Number.isFinite(v) ? "–" : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toPrecision(d);
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const PRIM_LABEL: Record<string, string> = { plane: "plane", cylinder: "cylinder", cone: "cone", sphere: "sphere", torus: "torus", freeform: "freeform" };
const STATUS_LABEL: Record<string, string> = {
  pending: "pending", running: "training", converged: "converged", finished: "max iterations", exact: "exact",
  stopped: "stopped", skipped: "skipped", error: "error",
};

export class App {
  private be = new Backend();
  private viewer = new Viewer($("viewer"));
  private loss = new LossChart($("chart-loss"));
  private pareto = new ParetoChart($("chart-pareto"));
  private regions = new Map<number, Region>();
  private slots = new Map<number, number>();
  private adjacency: [number, number][] = [];
  private focused: number | null = null;
  private active: number | null = null;
  private follow = true;
  private busy: string | null = null;
  private discovering = false;
  private stage = { mesh: false, cloud: false, regions: false, discovered: false };
  private meshName = "model";
  private diag = 1;
  private logLines: string[] = [];

  async start(): Promise<void> {
    this.bindControls();
    this.bindBackend();
    this.updateSteps();
    try {
      const info = await this.be.init();
      this.log(`backend started (python ${info.python}) · session ${info.workdir}${info.dev ? " · dev runtime" : ""}`);
    } catch (e) {
      this.toast("Could not start the backend", String(e));
      this.setEngine("error", "Backend failed to start");
    }
    if (!this.be.demo) {
      getCurrentWebview().onDragDropEvent((ev) => {
        if (ev.payload.type === "drop" && ev.payload.paths.length) this.loadMesh(ev.payload.paths[0]);
      });
    }
  }

  // ================================================================ controls
  private bindControls(): void {
    const outputs: [string, string, (v: number) => string][] = [
      ["in-angle", "out-angle", (v) => `${v}°`],
      ["in-tol", "out-tol", (v) => (v === 0 ? "auto" : `${v.toFixed(2)} %`)],
      ["in-ndev", "out-ndev", (v) => `${v}°`],
      ["in-minfrac", "out-minfrac", (v) => `${v.toFixed(2)} %`],
      ["in-maxsize", "out-maxsize", (v) => String(v)],
      ["in-iters", "out-iters", (v) => String(v)],
      ["in-conv", "out-conv", (v) => `${v.toFixed(2)} %`],
      ["in-trainpts", "out-trainpts", (v) => v.toLocaleString()],
      ["in-pops", "out-pops", (v) => String(v)],
    ];
    for (const [i, o, f] of outputs) {
      const inp = $<HTMLInputElement>(i);
      const upd = () => ($(o).textContent = f(Number(inp.value)));
      inp.addEventListener("input", upd);
      upd();
    }
    $("btn-open").addEventListener("click", async () => {
      if (this.be.demo) { this.loadMesh("demo/test_shapes.stl"); return; }
      const path = await open({
        multiple: false, directory: false, title: "Open mesh",
        filters: [{ name: "Meshes", extensions: ["stl", "obj", "ply", "off", "glb", "gltf", "fbx"] }],
      });
      if (typeof path === "string") this.loadMesh(path);
    });
    $("btn-segment").addEventListener("click", () =>
      this.cmd("segment", {
        angle_deg: Number($<HTMLInputElement>("in-angle").value),
        tol: Number($<HTMLInputElement>("in-tol").value) / 100,
        normal_deg: Number($<HTMLInputElement>("in-ndev").value),
        min_frac: Number($<HTMLInputElement>("in-minfrac").value) / 100,
        kinds: this.checked("prim-kinds"),
        strategy: $<HTMLSelectElement>("in-strategy").value,
      }));
    $("btn-discover").addEventListener("click", () => this.runDiscovery());
    $("btn-stop").addEventListener("click", () => this.be.send("stop"));
    $("btn-skip").addEventListener("click", () => this.be.send("skip"));
    $("btn-export").addEventListener("click", () => this.exportCad());
    $("btn-fit").addEventListener("click", () => this.viewer.fitView());
    $("btn-follow").addEventListener("click", () => {
      this.follow = true;
      if (this.active !== null) this.focus(this.active, false);
    });
    $("engine-pill").addEventListener("click", () => {
      if ($("engine-pill").dataset.state === "error") this.be.send("restart_worker");
    });
    $("in-frame").addEventListener("change", () => {
      if (this.focused === null) return;
      this.cmd("set_region_kind", { region: this.focused, kind: $<HTMLSelectElement>("in-frame").value });
    });
    // hole, boss and tube sizes in the region list are shown in the chosen units
    $("in-units").addEventListener("change", () => {
      for (const r of this.regions.values()) this.renderRegionRow(r);
    });
    $("layers").querySelectorAll("input").forEach((el) =>
      el.addEventListener("change", () => this.viewer.setLayer(el.value as any, el.checked)));
    $("colormode").querySelectorAll("input").forEach((el) =>
      el.addEventListener("change", () => {
        this.viewer.setColorMode(el.value as ColorMode);
        $("vp-legend").hidden = el.value !== "residual";
      }));
    $("btn-log").addEventListener("click", () => {
      const bar = $("logbar");
      const open = bar.dataset.open !== "true";
      bar.dataset.open = String(open);
      $("btn-log").setAttribute("aria-expanded", String(open));
      if (open) $("log").scrollTop = $("log").scrollHeight;
    });
    this.viewer.onPick = (rid) => { if (rid !== null && this.regions.has(rid)) this.focus(rid, true); else this.focus(null, true); };
    this.pareto.onSelect = (i) => {
      if (this.focused !== null) this.cmd("select_equation", { region: this.focused, index: i }, true);
    };
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      if (this.slots.size) { this.viewer.setSlots(this.slots); this.renderRegionList(); }
    });
  }

  private checked(id: string): string[] {
    return [...$(id).querySelectorAll<HTMLInputElement>("input:checked")].map((e) => e.value);
  }

  private async cmd(name: string, args: Record<string, unknown> = {}, allowWhileBusy = false): Promise<void> {
    if (this.busy && !allowWhileBusy) return;
    try {
      await this.be.send(name, args);
    } catch (e) {
      this.toast("Backend not reachable", String(e));
    }
  }

  private loadMesh(path: string): void {
    if (this.discovering) return;
    this.meshName = path.split(/[\\/]/).pop()!.replace(/\.[^.]+$/, "") || "model";
    this.cmd("load_mesh", { path });
  }

  private runDiscovery(): void {
    const scope = (document.querySelector<HTMLInputElement>('input[name="scope"]:checked')?.value) ?? "all";
    if (scope === "selected" && this.focused === null) {
      this.toast("No region selected", "Click a region in the list or in the 3D view first.");
      return;
    }
    const unary = this.checked("ops-unary");
    const binary = this.checked("ops-binary");
    if (!binary.length) { this.toast("Pick at least one operator", "Symbolic regression needs a binary operator."); return; }
    this.follow = true;
    this.cmd("discover", {
      regions: scope === "selected" ? [this.focused] : null,
      settings: {
        binary_operators: binary,
        unary_operators: unary,
        maxsize: Number($<HTMLInputElement>("in-maxsize").value),
        iterations: Number($<HTMLInputElement>("in-iters").value),
        tolerance: Number($<HTMLInputElement>("in-conv").value) / 100,
        max_points: Number($<HTMLInputElement>("in-trainpts").value),
        populations: Number($<HTMLInputElement>("in-pops").value),
      },
    });
  }

  private async exportCad(): Promise<void> {
    if (this.be.demo) { this.cmd("export"); return; }
    const dir = await open({ directory: true, multiple: false, title: "Choose export folder" });
    if (typeof dir !== "string") return;
    this.cmd("export", {
      out_dir: dir, name: this.meshName, formats: this.checked("export-formats"),
      units: $<HTMLSelectElement>("in-units").value,
    });
  }

  // ================================================================ backend events
  private bindBackend(): void {
    const be = this.be;
    be.onLog((line) => this.log(line));
    be.onExit(() => {
      this.setEngine("error", "Backend stopped");
      this.toast("Backend stopped", "Open the log for details. Click the engine status to restart the engine.");
    });
    be.on("hello", (e) => this.log(`backend ${e.version} ready (python ${e.python})`));
    be.on("julia_status", (e) => this.onEngine(e));
    be.on("log", (e) => this.log(e.msg, e.level));
    be.on("error", (e) => {
      this.toast(`${e.cmd ?? "Backend"} failed`, e.msg);
      this.log(`ERROR in ${e.cmd}: ${e.msg}\n${e.trace ?? ""}`, "error");
    });
    be.on("busy", (e) => {
      this.busy = e.busy ? e.cmd : null;
      if (!e.busy) $("vp-progress").hidden = true;
      this.updateSteps();
    });
    be.on("progress", (e) => {
      $("vp-progress").hidden = false;
      $("vp-progress-text").textContent = e.msg;
      $("vp-progress-bar").style.width = `${Math.round((e.frac ?? 0) * 100)}%`;
    });
    be.on("mesh_loaded", (e) => this.onMesh(e));
    be.on("cloud_ready", (e) => this.onCloud(e));
    be.on("segments_ready", (e) => this.onSegments(e));
    be.on("refine_level", (e) => this.onRefine(e));
    be.on("region_updated", (e) => this.onRegionUpdated(e));
    be.on("discover_start", () => { this.discovering = true; this.updateSteps(); });
    be.on("discover_done", (e) => {
      this.discovering = false;
      this.stage.discovered = [...this.regions.values()].some((r) => r.selected !== null);
      this.active = null;
      this.updateSteps();
      this.log(e.stopped ? "discovery stopped" : "discovery finished");
    });
    be.on("sr_start", (e) => this.onSrStart(e));
    be.on("sr_iter", (e) => this.onSrIter(e));
    be.on("sr_selected", (e) => this.onSrIter(e));
    be.on("sr_done", (e) => this.onSrDone(e));
    be.on("exported", (e) => this.onExported(e));
  }

  private onEngine(e: BackendEvent): void {
    const label: Record<string, string> = {
      loading: "Starting Julia…", compiling: "Compiling engine…", ready: "Engine ready",
      error: "Engine error · click to restart", stopped: "Engine stopped",
    };
    let text = label[e.state] ?? e.state;
    if (e.state === "ready" && e.threads) text += ` · ${e.threads} threads`;
    this.setEngine(e.state, text);
    this.log(`[engine] ${e.msg}${e.t ? ` (${Number(e.t).toFixed(1)} s)` : ""}`);
  }

  private setEngine(state: string, text: string): void {
    $("engine-pill").dataset.state = state;
    $("engine-text").textContent = text;
  }

  private async onMesh(e: BackendEvent): Promise<void> {
    const [pos, idx] = await Promise.all([this.be.readF32(e.files.positions), this.be.readU32(e.files.indices)]);
    this.viewer.setMesh(pos, idx);
    this.regions.clear();
    this.focus(null, false);
    this.renderRegionList();
    // the mesh's own vertices are the points: nothing to sample
    this.stage = { mesh: true, cloud: true, regions: false, discovered: false };
    const s = e.stats;
    $("pts-info").textContent = s.cad
      ? `CAD export detected: its ${s.n_vertices.toLocaleString()} vertices lie exactly on the surfaces and its triangles are segmented directly.`
      : `${s.n_vertices.toLocaleString()} mesh vertices are used as the points.`;
    // On a CAD mesh the points are its own vertices: the shaded mesh, coloured by
    // region, shows the surfaces more clearly than thousands of dots over it.
    const pts = document.querySelector<HTMLInputElement>('#layers input[value="points"]');
    if (pts && pts.checked === !!s.cad) {
      pts.checked = !s.cad;
      this.viewer.setLayer("points", !s.cad);
    }
    $("levels").hidden = true;
    $("levels").innerHTML = "";
    this.diag = s.diag;
    if (s.units_hint) $<HTMLSelectElement>("in-units").value = s.units_hint;
    $("vp-empty").hidden = true;
    $("file-name").textContent = e.name;
    $("mesh-stats").innerHTML = [
      ["Vertices", s.n_vertices.toLocaleString()],
      ["Triangles", s.n_faces.toLocaleString()],
      ["Size", s.bbox_max.map((v: number, i: number) => fmt(v - s.bbox_min[i], 4)).join(" × ")],
      ["Watertight", s.watertight ? "yes" : "no"],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    $("export-files").innerHTML = "";
    this.updateSteps();
  }

  private async onCloud(e: BackendEvent): Promise<void> {
    const [pos, nrm] = await Promise.all([this.be.readF32(e.files.positions), this.be.readF32(e.files.normals)]);
    this.viewer.setPoints(pos, nrm);
    this.regions.clear();
    this.focus(null, false);
    this.renderRegionList();
    this.stage.cloud = true;
    this.stage.regions = false;
    this.stage.discovered = false;
    this.log(`point cloud: ${e.n_points.toLocaleString()} points`);
    this.updateSteps();
  }

  /** Coarse-to-fine: show each level as it is found (overall shape -> directions -> planes -> details). */
  private async onRefine(e: BackendEvent): Promise<void> {
    const ol = $("levels");
    ol.hidden = false;
    if (e.level === 0) ol.innerHTML = "";
    ol.querySelectorAll("li").forEach((li) => ((li as HTMLElement).dataset.active = "false"));
    const li = document.createElement("li");
    li.dataset.active = "true";
    li.innerHTML = "<b><span class=\"lv\">L" + e.level + "</span>" + esc(e.name) + "</b><span>" + esc(e.summary ?? "") + "</span>";
    if (e.latex) {
      const d = document.createElement("div");
      d.className = "eq";
      li.appendChild(d);
      this.tex(d, e.latex);
    }
    ol.appendChild(li);
    this.log("L" + e.level + " " + e.name + ": " + (e.summary ?? "") + (e.expression ? " | " + e.expression : ""));
    if (e.preview) this.viewer.setOverlay(b64F32(e.preview.positions), b64U32(e.preview.indices));
    if (e.files?.face_labels) this.viewer.showFaceLevel(await this.be.readU32(e.files.face_labels));
  }

  private async onSegments(e: BackendEvent): Promise<void> {
    const [labels, faceLabels] = await Promise.all([this.be.readU32(e.files.labels), this.be.readU32(e.files.face_labels)]);
    this.regions.clear();
    for (const r of e.regions) this.regions.set(r.id, { ...r, hof: [], selected: null, history: [] });
    this.adjacency = e.adjacency;
    this.slots = regionSlots([...this.regions.keys()], this.adjacency);
    this.viewer.endLevelView();
    this.viewer.setLabels(labels, faceLabels, this.slots);
    this.viewer.residualScale = e.tolerance ?? (Number($<HTMLInputElement>("in-tol").value) / 100) * this.diag;
    $("lg-min").textContent = `−${fmt(this.viewer.residualScale, 2)}`;
    $("lg-max").textContent = `+${fmt(this.viewer.residualScale, 2)}`;
    this.stage.regions = true;
    // A CAD mesh's planes, cylinders, cones, spheres and tori come with their exact
    // equations; freeform faces export as fitted B-splines. Then discovery is optional
    // (it finds formulas for the freeform faces) and Export is the next step.
    const exact = e.regions.filter((r: any) => r.status === "exact").length;
    // freeform faces still to fit (a thread flank found as an exact helical sweep is not one)
    const freeform = e.regions.filter((r: any) => r.prim === "freeform" && r.status !== "exact").length;
    const tubes = e.regions.filter((r: any) => r.prim === "freeform" && r.params?.tube_r).length;
    const threads = e.regions.filter((r: any) => r.classification?.type === "helix").length;
    this.stage.discovered = e.regions.length > 0 && exact + freeform === e.regions.length && exact > 0;
    this.focus(null, false);
    this.renderRegionList();
    const counts: Record<string, number> = {};
    for (const r of e.regions) counts[r.prim] = (counts[r.prim] ?? 0) + 1;
    this.log(`regions: ${e.regions.length} (${Object.entries(counts).map(([k, v]) => `${v} ${k}`).join(", ")}) in ${e.t.toFixed(1)} s`);
    if (this.stage.discovered)
      this.log(`${exact} surfaces have exact equations${freeform ? `; ${freeform} freeform export as fitted B-splines (Discover finds formulas for them)` : ""}` +
        `${tubes ? `; ${tubes} of them are swept tubes (springs, wires), each one face` : ""}` +
        `${threads ? `; ${threads} thread flanks are exact helical sweeps` : ""}. Ready to export.`);
    this.updateSteps();
  }

  private onRegionUpdated(e: BackendEvent): void {
    const r = e.region;
    this.regions.set(r.id, { ...r, hof: [], selected: null, history: [] });
    this.viewer.removeSurface(r.id);
    this.viewer.clearResiduals(r.id);
    this.renderRegionList();
    if (this.focused === r.id) this.renderMonitor();
  }

  private onSrStart(e: BackendEvent): void {
    const r = this.regions.get(e.region);
    if (!r) return;
    Object.assign(r, { status: "running", hof: [], selected: null, history: [], iteration: 0,
      maxIter: e.max_iterations, targetLoss: e.target_loss, nTrain: e.n_train, frame: e.frame, guesses: e.guesses });
    this.active = e.region;
    if (this.follow) this.focus(e.region, false);
    this.renderRegionRow(r);
    if (this.focused === r.id) this.renderMonitor();
  }

  private onSrIter(e: BackendEvent): void {
    const r = this.regions.get(e.region);
    if (!r) return;
    if (e.hof) r.hof = e.hof;
    if (e.type === "sr_iter") {
      r.iteration = e.iteration;
      r.dt = e.dt;
      r.history.push(e.history);
    }
    Object.assign(r, {
      selected: e.selected, latex: e.latex, world: e.world, world_latex: e.world_latex,
      rms: e.rms, rms_rel: e.rms_rel, max_err: e.max_err, classification: e.classification,
    });
    if (e.summary?.status) r.status = e.summary.status;
    this.viewer.setSurface(r.id, b64F32(e.preview.positions), b64U32(e.preview.indices));
    this.viewer.setResiduals(r.id, b64F32(e.residuals));
    this.renderRegionRow(r);
    if (this.focused === r.id) this.renderMonitor();
  }

  private onSrDone(e: BackendEvent): void {
    const r = this.regions.get(e.region);
    if (!r) return;
    r.status = e.status;
    if (e.summary?.classification) r.classification = e.summary.classification;
    this.renderRegionRow(r);
    if (this.focused === r.id) this.renderMonitor();
  }

  private onExported(e: BackendEvent): void {
    const labels: Record<string, string> = {
      step: "STEP (B-rep)", snapped_stl: "Snapped mesh (STL)", equation_stl: "Mesh from equations (STL)", json: "Equations (JSON)",
    };
    const ul = $("export-files");
    ul.innerHTML = "";
    for (const [k, path] of Object.entries<string>(e.files)) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="name" title="${esc(path)}">${labels[k] ?? k}</span>`;
      const b = document.createElement("button");
      b.className = "btn small";
      b.textContent = "Show";
      b.onclick = () => this.be.reveal(path);
      li.appendChild(b);
      ul.appendChild(li);
    }
    const st = e.info?.step;
    const solids = st && st.parts > 1
      ? `${st.parts} parts, ${st.closed_parts} closed solids`
      : st ? (st.closed ? "closed solid" : "open shell") : "";
    this.log(`exported to ${e.out_dir}${st ? ` · STEP: ${st.faces} faces, ${solids}` : ""}`);
    for (const w of st?.warnings ?? []) this.log(w, "warn");
  }

  // ================================================================ rendering
  private updateSteps(): void {
    const set = (step: string, done: boolean, locked: boolean) => {
      const el = document.querySelector<HTMLElement>(`.step[data-step="${step}"]`)!;
      el.dataset.done = String(done);
      el.dataset.locked = String(locked);
    };
    const s = this.stage;
    const busy = !!this.busy;
    set("mesh", s.mesh, this.discovering);
    set("cloud", s.cloud, !s.mesh || this.discovering);
    set("regions", s.regions, !s.mesh || this.discovering);
    set("discover", s.discovered, !s.regions);
    set("cad", false, !s.regions || this.discovering);
    for (const id of ["btn-open", "btn-segment", "btn-discover", "btn-export"])
      $<HTMLButtonElement>(id).disabled = busy;
    $<HTMLButtonElement>("btn-stop").disabled = !this.discovering;
    $<HTMLButtonElement>("btn-skip").disabled = !this.discovering;
    $<HTMLSelectElement>("in-frame").disabled = this.focused === null || this.discovering || busy;
    // the next step's button is the primary one
    const next = !s.mesh ? "btn-open" : !s.regions ? "btn-segment" : !s.discovered ? "btn-discover" : "btn-export";
    for (const id of ["btn-open", "btn-segment", "btn-discover", "btn-export"])
      $(id).classList.toggle("primary", id === next);
  }

  private swatch(rid: number): string {
    return categorical()[this.slots.get(rid) ?? 0];
  }

  private shortEq(r: Region): string {
    if (r.world) return r.world;
    return "";
  }

  private renderRegionList(): void {
    const list = $("region-list");
    list.innerHTML = "";
    $("regions-count").textContent = this.regions.size ? String(this.regions.size) : "";
    if (!this.regions.size) {
      list.innerHTML = `<div class="empty">No regions yet</div>`;
      return;
    }
    for (const r of this.regions.values()) {
      const row = document.createElement("div");
      row.className = "rrow";
      row.id = `rrow-${r.id}`;
      row.setAttribute("role", "option");
      row.onclick = () => this.focus(this.focused === r.id ? null : r.id, true);
      list.appendChild(row);
      this.renderRegionRow(r);
    }
  }

  private renderRegionRow(r: Region): void {
    const row = document.getElementById(`rrow-${r.id}`);
    if (!row) return;
    const cls = r.classification?.type;
    const type = cls && cls !== "freeform" ? (cls === "helix" ? "thread" : cls) : PRIM_LABEL[r.prim] ?? r.prim;
    // a coil spring or bent wire: one swept tube surface, labelled by its wire size
    const tubeR = r.prim === "freeform" ? r.params?.tube_r : undefined;
    const kind = r.feature ? `${r.feature.kind} Ø${this.lengthText(r.feature.diameter)}`
      : tubeR ? `tube Ø${this.lengthText(2 * tubeR)}` : type;
    row.setAttribute("aria-selected", String(this.focused === r.id));
    row.title = `${r.n_points.toLocaleString()} points · ${r.frame.kind} frame`;
    row.innerHTML =
      `<span class="sw" style="background:${this.swatch(r.id)}"></span>` +
      `<span class="id">R${r.id}</span>` +
      `<span class="kind">${esc(kind)}</span>` +
      `<span class="eqs">${esc(this.shortEq(r))}</span>` +
      `<span class="badge" data-s="${r.status}">${r.status === "pending" ? "" : STATUS_LABEL[r.status] ?? r.status}</span>`;
  }

  /** A model-unit length in mm (from the chosen model units), e.g. "14.50 mm". */
  private lengthText(v: number): string {
    const mm = { mm: 1, cm: 10, m: 1000, inch: 25.4 }[$<HTMLSelectElement>("in-units").value] ?? 1;
    return `${(v * mm).toFixed(2)} mm`;
  }

  private focus(rid: number | null, user: boolean): void {
    if (user && this.discovering && rid !== this.active) this.follow = false;
    if (user && rid === this.active) this.follow = true;
    const prev = this.focused;
    this.focused = rid;
    this.viewer.setHighlight(rid);
    if (prev !== null) { const p = this.regions.get(prev); if (p) this.renderRegionRow(p); }
    if (rid !== null) {
      const r = this.regions.get(rid);
      if (r) this.renderRegionRow(r);
      document.getElementById(`rrow-${rid}`)?.scrollIntoView({ block: "nearest" });
    }
    $("btn-follow").hidden = !(this.discovering && !this.follow);
    this.updateSteps();
    this.renderMonitor();
  }

  private renderMonitor(): void {
    const r = this.focused !== null ? this.regions.get(this.focused) : undefined;
    $("btn-follow").hidden = !(this.discovering && !this.follow);
    if (!r) {
      $("mon-title").textContent = "Training monitor";
      $("mon-iter").textContent = this.regions.size ? "select a region" : "";
      $("mon-status").textContent = "";
      $("eq-world").innerHTML = `<span class="muted">—</span>`;
      $("eq-norm").innerHTML = `<span class="muted">—</span>`;
      $("eq-meta").innerHTML = "";
      this.loss.setData([], 10, null);
      this.pareto.setData([], null, Number($<HTMLInputElement>("in-maxsize").value), null);
      $("hof-table").querySelector("tbody")!.innerHTML = "";
      return;
    }
    const ct = r.classification?.type;
    const type = ct && ct !== "freeform" ? (ct === "helix" ? "thread" : ct) : r.prim;
    $("mon-title").textContent = `Region ${r.id} · ${type}`;
    $("mon-iter").textContent = r.iteration
      ? `iteration ${r.iteration} / ${r.maxIter}${r.dt ? ` · ${r.dt.toFixed(2)} s/it` : ""}`
      : r.status === "running" ? "starting…" : "";
    const badge = $("mon-status");
    badge.dataset.s = r.status;
    badge.textContent = r.status === "pending" ? "not searched" : STATUS_LABEL[r.status] ?? r.status;
    $<HTMLSelectElement>("in-frame").value = r.frame.kind;

    if (r.world_latex) this.tex($("eq-world"), r.world_latex);
    else $("eq-world").innerHTML = `<span class="muted">${r.status === "running" ? "searching…" : "run discovery to find an equation"}</span>`;
    const yname = r.frame.kind === "height" ? "\\hat{w}" : "\\hat{r}";
    if (r.latex) this.tex($("eq-norm"), `${yname} = ${r.latex}`);
    else $("eq-norm").innerHTML = `<span class="muted">—</span>`;

    const vars = r.frame.kind === "height" ? "u, v ⟂ w" : r.frame.kind === "cylindrical" ? "θ around axis, h along axis" : "θ, latitude about centre";
    const ax = r.frame.e3.map((v) => fmt(v, 3)).join(", ");
    const org = r.frame.origin.map((v) => fmt(v, 4)).join(", ");
    const meta: [string, string][] = [
      ["fit error", r.rms != null ? `${fmt(r.rms)} rms · ${fmt(r.max_err)} max (${fmt((r.rms_rel ?? 0) * 100, 2)} % of size)` : "–"],
      ["frame", `${r.frame.kind} · ${vars}`],
      [r.frame.kind === "height" ? "normal" : r.frame.kind === "cylindrical" ? "axis" : "pole", `(${ax})`],
      ["origin", `(${org})`],
      ["points", `${r.n_points.toLocaleString()}${r.nTrain ? ` · ${r.nTrain.toLocaleString()} used for training` : ""}`],
    ];
    if (r.classification?.type) meta.splice(1, 0, ["CAD surface", this.describeCls(r.classification)]);
    if (r.guesses?.length) meta.push(["seeded with", `${r.guesses.join(", ")} (from region fit)`]);
    $("eq-meta").innerHTML = meta.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");

    this.loss.setData(r.history, r.maxIter ?? Number($<HTMLInputElement>("in-iters").value), r.targetLoss ?? null);
    this.pareto.setData(r.hof, r.selected, Number($<HTMLInputElement>("in-maxsize").value), r.targetLoss ?? null);
    const tbody = $("hof-table").querySelector("tbody")!;
    tbody.innerHTML = "";
    r.hof.forEach((h, i) => {
      const tr = document.createElement("tr");
      tr.setAttribute("aria-selected", String(i === r.selected));
      tr.innerHTML = `<td>${h.complexity}</td><td>${fmtLoss(h.loss)}</td><td title="${esc(h.equation)}">${esc(h.equation)}</td>`;
      tr.onclick = () => this.cmd("select_equation", { region: r.id, index: i }, true);
      tbody.appendChild(tr);
    });
  }

  private describeCls(c: { type: string; [k: string]: any }): string {
    switch (c.type) {
      case "plane": return "plane (exact)";
      case "cylinder": return `cylinder, radius ${fmt(c.radius, 5)}`;
      case "sphere": return `sphere, radius ${fmt(c.radius, 5)}`;
      case "cone": return `cone, half-angle ${fmt((Math.atan(Math.abs(c.slope)) * 180) / Math.PI, 4)}°`;
      case "torus": return `torus, radii ${fmt(c.R, 5)} / ${fmt(c.r, 5)}`;
      case "revolve": return "surface of revolution (B-spline)";
      case "helix": return `thread flank: helical sweep, lead ${fmt(Math.abs(c.lead), 5)} (${c.lead > 0 ? "right" : "left"}-handed)`;
      default: return "freeform (B-spline from equation)";
    }
  }

  private tex(el: HTMLElement, latex: string): void {
    try {
      katex.render(latex, el, { throwOnError: false, displayMode: false });
    } catch {
      el.textContent = latex;
    }
  }

  // ================================================================ misc UI
  private log(msg: string, level: string = "info"): void {
    if (!msg) return;
    const lvl = level === "error" || /error|traceback/i.test(msg.slice(0, 40)) ? "err" : level === "warn" ? "warn" : "";
    const time = new Date().toLocaleTimeString();
    this.logLines.push(`<span class="${lvl}">${esc(`${time}  ${msg}`)}</span>`);
    if (this.logLines.length > 3000) this.logLines.splice(0, 1000);
    $("log-last").textContent = msg.split("\n")[0];
    const log = $("log");
    if ($("logbar").dataset.open === "true") {
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
      log.innerHTML = this.logLines.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
    } else {
      this.logDirty = true;
    }
  }
  private logDirty = false;

  toast(title: string, body: string): void {
    const t = document.createElement("div");
    t.className = "toast";
    t.innerHTML = `<b>${esc(title)}</b>${esc(body).slice(0, 400)}`;
    $("toasts").appendChild(t);
    setTimeout(() => t.remove(), 8000);
  }

  flushLog(): void {
    if (this.logDirty) {
      $("log").innerHTML = this.logLines.join("\n");
      this.logDirty = false;
    }
  }
}
