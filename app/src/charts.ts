// Small canvas charts for the training monitor (log-scale loss).
// Marks: 2px lines, >=8px markers with a 2px surface ring, recessive grid,
// legend for >= 2 series, crosshair/nearest-point tooltip on hover.

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const pow10 = (e: number) => "10" + String(e).split("").map((c) => SUP[c] ?? c).join("");
const FLOOR = 1e-12;

export function fmtLoss(v: number): string {
  if (!Number.isFinite(v)) return "–";
  if (v <= FLOOR) return "0";
  return v >= 1e-3 && v < 1e4 ? v.toPrecision(3) : v.toExponential(2);
}

function logDomain(vals: number[]): [number, number] {
  const pos = vals.filter((v) => Number.isFinite(v)).map((v) => Math.max(v, FLOOR));
  if (!pos.length) return [-6, 0];
  let lo = Math.floor(Math.log10(Math.min(...pos)));
  let hi = Math.ceil(Math.log10(Math.max(...pos)));
  if (hi - lo < 2) { lo -= 1; hi += 1; }
  return [lo, hi];
}

abstract class CanvasChart {
  protected canvas = document.createElement("canvas");
  protected ctx: CanvasRenderingContext2D;
  protected tip = document.createElement("div");
  protected w = 0;
  protected h = 0;
  protected m = { l: 46, r: 12, t: 22, b: 22 };

  constructor(protected el: HTMLElement) {
    el.appendChild(this.canvas);
    this.tip.className = "tip";
    this.tip.hidden = true;
    el.appendChild(this.tip);
    this.ctx = this.canvas.getContext("2d")!;
    new ResizeObserver(() => { this.resize(); this.draw(); }).observe(el);
    this.canvas.addEventListener("mousemove", (e) => this.hover(e.offsetX, e.offsetY));
    this.canvas.addEventListener("mouseleave", () => { this.tip.hidden = true; this.draw(); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
    this.resize();
  }

  protected resize(): void {
    const dpr = window.devicePixelRatio || 1;
    this.w = this.el.clientWidth;
    this.h = this.el.clientHeight;
    this.canvas.width = Math.max(1, this.w * dpr);
    this.canvas.height = Math.max(1, this.h * dpr);
    this.canvas.style.width = this.w + "px";
    this.canvas.style.height = this.h + "px";
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  protected abstract draw(): void;
  protected abstract hover(x: number, y: number): void;

  protected yAxis(lo: number, hi: number, y: (e: number) => number): void {
    const g = this.ctx;
    const step = hi - lo > 8 ? 2 : 1;
    g.font = `11px ${css("--font")}`;
    g.textAlign = "right";
    g.textBaseline = "middle";
    for (let e = lo; e <= hi; e += step) {
      const yy = Math.round(y(e)) + 0.5;
      g.strokeStyle = css("--border");
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(this.m.l, yy); g.lineTo(this.w - this.m.r, yy); g.stroke();
      g.fillStyle = css("--text-muted");
      g.fillText(pow10(e), this.m.l - 6, yy);
    }
  }

  protected xTicks(lo: number, hi: number, x: (v: number) => number): void {
    const g = this.ctx;
    const span = hi - lo;
    const step = [1, 2, 5, 10, 20, 25, 50, 100].find((s) => span / s <= 8) ?? 200;
    g.textAlign = "center";
    g.textBaseline = "top";
    g.fillStyle = css("--text-muted");
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
      if (v < lo) continue;
      g.fillText(String(v), x(v), this.h - this.m.b + 6);
    }
    g.strokeStyle = css("--border-strong");
    g.beginPath();
    g.moveTo(this.m.l, this.h - this.m.b + 0.5);
    g.lineTo(this.w - this.m.r, this.h - this.m.b + 0.5);
    g.stroke();
  }

  protected refLine(yy: number, label: string): void {
    const g = this.ctx;
    g.save();
    g.setLineDash([4, 4]);
    g.strokeStyle = css("--text-muted");
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(this.m.l, Math.round(yy) + 0.5); g.lineTo(this.w - this.m.r, Math.round(yy) + 0.5); g.stroke();
    g.restore();
    g.fillStyle = css("--text-muted");
    g.textAlign = "right";
    g.textBaseline = "bottom";
    g.fillText(label, this.w - this.m.r - 2, yy - 2);
  }

  protected marker(x: number, y: number, r: number, fill: string, ring = css("--panel-2")): void {
    const g = this.ctx;
    g.beginPath();
    g.arc(x, y, r + 2, 0, Math.PI * 2);
    g.fillStyle = ring;
    g.fill();
    g.beginPath();
    g.arc(x, y, r, 0, Math.PI * 2);
    g.fillStyle = fill;
    g.fill();
  }

  protected showTip(x: number, y: number, html: string): void {
    this.tip.innerHTML = html;
    this.tip.hidden = false;
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    let left = x + 12, top = y - th - 8;
    if (left + tw > this.w) left = x - tw - 12;
    if (top < 0) top = y + 12;
    this.tip.style.left = Math.max(0, left) + "px";
    this.tip.style.top = top + "px";
  }
}

// ------------------------------------------------------------------ loss per iteration

export interface HistPoint { iteration: number; best_loss: number; selected_loss: number }

export class LossChart extends CanvasChart {
  private hist: HistPoint[] = [];
  private maxIter = 10;
  private target: number | null = null;
  private hoverIt: number | null = null;

  setData(hist: HistPoint[], maxIter: number, target: number | null): void {
    this.hist = hist;
    this.maxIter = Math.max(maxIter, 2);
    this.target = target;
    this.draw();
  }

  private scales() {
    const vals = this.hist.flatMap((h) => [h.best_loss, h.selected_loss]);
    if (this.target) vals.push(this.target);
    const [lo, hi] = logDomain(vals);
    const x = (v: number) => this.m.l + ((v - 1) / (this.maxIter - 1)) * (this.w - this.m.l - this.m.r);
    const y = (e: number) => this.m.t + ((hi - e) / (hi - lo)) * (this.h - this.m.t - this.m.b);
    const ly = (v: number) => y(Math.log10(Math.max(v, FLOOR)));
    return { lo, hi, x, y, ly };
  }

  protected draw(): void {
    const g = this.ctx;
    g.clearRect(0, 0, this.w, this.h);
    if (this.w < 50) return;
    const { lo, hi, x, y, ly } = this.scales();
    this.yAxis(lo, hi, y);
    this.xTicks(1, this.maxIter, x);
    if (this.target && this.hist.length) this.refLine(ly(this.target), "converged below");
    const series = [
      { key: "best_loss" as const, name: "best loss", color: css("--series-1") },
      { key: "selected_loss" as const, name: "chosen equation", color: css("--series-2") },
    ];
    // legend
    g.font = `11px ${css("--font")}`;
    g.textBaseline = "middle";
    g.textAlign = "left";
    let lx = this.m.l;
    for (const s of series) {
      g.fillStyle = s.color;
      g.fillRect(lx, 6, 10, 3);
      g.fillStyle = css("--text-secondary");
      g.fillText(s.name, lx + 14, 8);
      lx += g.measureText(s.name).width + 30;
    }
    if (!this.hist.length) {
      g.fillStyle = css("--text-muted");
      g.textAlign = "center";
      g.fillText("waiting for the first iteration", (this.m.l + this.w) / 2, this.h / 2);
      return;
    }
    for (const s of series) {
      g.strokeStyle = s.color;
      g.lineWidth = 2;
      g.lineJoin = "round";
      g.beginPath();
      this.hist.forEach((h, i) => (i ? g.lineTo(x(h.iteration), ly(h[s.key])) : g.moveTo(x(h.iteration), ly(h[s.key]))));
      g.stroke();
      const last = this.hist[this.hist.length - 1];
      this.marker(x(last.iteration), ly(last[s.key]), 3.5, s.color);
    }
    if (this.hoverIt !== null) {
      const h = this.hist.find((p) => p.iteration === this.hoverIt);
      if (h) {
        g.strokeStyle = css("--text-muted");
        g.lineWidth = 1;
        g.beginPath(); g.moveTo(Math.round(x(h.iteration)) + 0.5, this.m.t); g.lineTo(Math.round(x(h.iteration)) + 0.5, this.h - this.m.b); g.stroke();
        for (const s of series) this.marker(x(h.iteration), ly(h[s.key]), 4, s.color);
      }
    }
  }

  protected hover(px: number, py: number): void {
    if (!this.hist.length) return;
    const { x } = this.scales();
    let best = this.hist[0];
    for (const h of this.hist) if (Math.abs(x(h.iteration) - px) < Math.abs(x(best.iteration) - px)) best = h;
    this.hoverIt = best.iteration;
    this.draw();
    this.showTip(px, py,
      `<div><span class="k">iteration</span> ${best.iteration}</div>` +
      `<div><span class="sw" style="background:${css("--series-1")}"></span>best loss ${fmtLoss(best.best_loss)}</div>` +
      `<div><span class="sw" style="background:${css("--series-2")}"></span>chosen ${fmtLoss(best.selected_loss)}</div>`);
  }
}

// ------------------------------------------------------------------ Pareto front

export interface HofRow { complexity: number; loss: number; equation: string }

export class ParetoChart extends CanvasChart {
  private rows: HofRow[] = [];
  private selected: number | null = null;
  private target: number | null = null;
  private maxC = 20;
  private hoverIdx: number | null = null;
  onSelect: ((i: number) => void) | null = null;

  constructor(el: HTMLElement) {
    super(el);
    this.canvas.addEventListener("click", () => {
      if (this.hoverIdx !== null) this.onSelect?.(this.hoverIdx);
    });
  }

  setData(rows: HofRow[], selected: number | null, maxC: number, target: number | null): void {
    this.rows = rows;
    this.selected = selected;
    this.maxC = Math.max(maxC, ...rows.map((r) => r.complexity), 2);
    this.target = target;
    this.draw();
  }

  private scales() {
    const vals = this.rows.map((r) => r.loss);
    if (this.target) vals.push(this.target);
    const [lo, hi] = logDomain(vals);
    const x = (c: number) => this.m.l + ((c - 0.5) / (this.maxC)) * (this.w - this.m.l - this.m.r);
    const y = (e: number) => this.m.t + ((hi - e) / (hi - lo)) * (this.h - this.m.t - this.m.b);
    const ly = (v: number) => y(Math.log10(Math.max(v, FLOOR)));
    return { lo, hi, x, y, ly };
  }

  protected draw(): void {
    const g = this.ctx;
    g.clearRect(0, 0, this.w, this.h);
    if (this.w < 50) return;
    const { lo, hi, x, y, ly } = this.scales();
    this.yAxis(lo, hi, y);
    this.xTicks(1, this.maxC, x);
    g.fillStyle = css("--text-muted");
    if (this.target && this.rows.length) this.refLine(ly(this.target), "tolerance");
    if (!this.rows.length) {
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText("no equations yet", (this.m.l + this.w) / 2, this.h / 2);
      return;
    }
    const color = css("--series-1");
    const order = this.rows.map((_, i) => i).sort((a, b) => this.rows[a].complexity - this.rows[b].complexity);
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.beginPath();
    order.forEach((i, k) => {
      const r = this.rows[i];
      if (!k) { g.moveTo(x(r.complexity), ly(r.loss)); return; }
      g.lineTo(x(r.complexity), ly(this.rows[order[k - 1]].loss));
      g.lineTo(x(r.complexity), ly(r.loss));
    });
    g.stroke();
    order.forEach((i) => {
      const r = this.rows[i];
      const sel = i === this.selected, hov = i === this.hoverIdx;
      this.marker(x(r.complexity), ly(r.loss), sel ? 5.5 : hov ? 5 : 4, color, sel ? css("--text-primary") : css("--panel-2"));
    });
  }

  protected hover(px: number, py: number): void {
    const { x, ly } = this.scales();
    let best: number | null = null, bd = 196;
    this.rows.forEach((r, i) => {
      const d = (x(r.complexity) - px) ** 2 + (ly(r.loss) - py) ** 2;
      if (d < bd) { bd = d; best = i; }
    });
    this.hoverIdx = best;
    this.canvas.style.cursor = best !== null ? "pointer" : "default";
    this.draw();
    if (best === null) { this.tip.hidden = true; return; }
    const r = this.rows[best];
    const eq = r.equation.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    this.showTip(px, py, `<div><span class="k">complexity</span> ${r.complexity} · <span class="k">loss</span> ${fmtLoss(r.loss)}</div><div style="font-family:var(--mono)">${eq}</div>`);
  }
}
