// Thin wrapper around the Tauri commands/events that talk to the Python backend.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

export interface RuntimeInfo {
  runtime: string;
  backend: string;
  workdir: string;
  python: string;
  dev: boolean;
}

export type BackendEvent = { type: string; [k: string]: any };
type Handler = (e: BackendEvent) => void;

export const isTauri = () => "__TAURI_INTERNALS__" in window;

export class Backend {
  private handlers = new Map<string, Set<Handler>>();
  private logHandlers = new Set<(line: string) => void>();
  private exitHandlers = new Set<() => void>();
  info: RuntimeInfo | null = null;
  /** Browser-only replay of a recorded session (dev preview, never in the app). */
  demo: DemoReplay | null = null;

  async init(): Promise<RuntimeInfo> {
    if (!isTauri()) {
      this.demo = new DemoReplay((e) => this.dispatch(e));
      await this.demo.load();
      this.info = { runtime: "demo", backend: "demo", workdir: "demo", python: "demo", dev: true };
      return this.info;
    }
    await listen<BackendEvent>("backend-event", (e) => this.dispatch(e.payload));
    await listen<string>("backend-log", (e) => this.logHandlers.forEach((h) => h(e.payload)));
    await listen("backend-exit", () => this.exitHandlers.forEach((h) => h()));
    this.info = await invoke<RuntimeInfo>("backend_start");
    return this.info;
  }

  async restart(): Promise<RuntimeInfo> {
    this.info = await invoke<RuntimeInfo>("backend_restart");
    return this.info;
  }

  on(type: string, h: Handler): void {
    if (!this.handlers.has(type)) this.handlers.set(type, new Set());
    this.handlers.get(type)!.add(h);
  }
  onLog(h: (line: string) => void): void { this.logHandlers.add(h); }
  onExit(h: () => void): void { this.exitHandlers.add(h); }

  private dispatch(e: BackendEvent): void {
    this.handlers.get(e.type)?.forEach((h) => h(e));
    this.handlers.get("*")?.forEach((h) => h(e));
  }

  send(cmd: string, args: Record<string, unknown> = {}): Promise<void> {
    if (this.demo) return this.demo.command(cmd);
    return invoke("backend_send", { msg: { cmd, args } });
  }

  async readBytes(path: string): Promise<ArrayBuffer> {
    if (this.demo) return (await fetch(path)).arrayBuffer();
    return invoke<ArrayBuffer>("read_binary", { path });
  }
  async readF32(path: string): Promise<Float32Array> { return new Float32Array(await this.readBytes(path)); }
  async readU32(path: string): Promise<Uint32Array> { return new Uint32Array(await this.readBytes(path)); }

  reveal(path: string): Promise<void> { return invoke("reveal_path", { path }); }
}

/** Replays events recorded by scripts/record_demo.py, grouped by command. */
class DemoReplay {
  private groups: Record<string, { dt: number; ev: BackendEvent }[]> = {};
  private stopped = false;
  constructor(private emit: (e: BackendEvent) => void) {}

  async load(): Promise<void> {
    try {
      this.groups = await (await fetch("/demo/events.json")).json();
    } catch {
      this.groups = {};
    }
    setTimeout(() => this.play("_boot"), 50);
  }

  async command(cmd: string): Promise<void> {
    if (cmd === "stop") { this.stopped = true; return; }
    if (cmd === "skip") return;
    this.stopped = false;
    this.emit({ type: "busy", busy: true, cmd });
    await this.play(cmd);
    this.emit({ type: "busy", busy: false, cmd });
  }

  private async play(cmd: string): Promise<void> {
    for (const { dt, ev } of this.groups[cmd] ?? []) {
      if (this.stopped && cmd === "discover") break;
      await new Promise((r) => setTimeout(r, Math.min(dt * 1000, 1500)));
      this.emit(ev);
    }
  }
}

export function b64Bytes(s: string): ArrayBuffer {
  const bin = atob(s);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8.buffer;
}
export const b64F32 = (s: string) => new Float32Array(b64Bytes(s));
export const b64U32 = (s: string) => new Uint32Array(b64Bytes(s));
