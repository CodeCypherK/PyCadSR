import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { categorical, diverging, NEUTRAL } from "./palette";

export type ColorMode = "region" | "residual" | "normal";
type RGB = [number, number, number];

function linear([r, g, b]: RGB): THREE.Color {
  return new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
}
function hexRGB(hex: string): RGB {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function discTexture(): THREE.Texture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  g.beginPath();
  g.arc(32, 32, 28, 0, Math.PI * 2);
  g.fillStyle = "#fff";
  g.fill();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Viewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private meshObj: THREE.Mesh | null = null;
  private pointsObj: THREE.Points | null = null;
  private surfaceGroup = new THREE.Group();
  private surfaces = new Map<number, THREE.Mesh>();
  private labels: Uint32Array | null = null;
  private faceLabels: Uint32Array | null = null;
  private normals: Float32Array | null = null;
  private pointRes: Float32Array | null = null;
  private regionIdx = new Map<number, Uint32Array>();
  private regionRGB = new Map<number, RGB>();
  private layers = { mesh: true, points: true, surfaces: true };
  private highlight: number | null = null;
  private needsRender = true;
  private diag = 1;
  private disc = discTexture();
  private overlay: THREE.Group | null = null;
  private levelView = false;
  colorMode: ColorMode = "region";
  residualScale = 1;
  onPick: ((rid: number | null) => void) | null = null;

  constructor(private el: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1e7);
    this.camera.up.set(0, 0, 1); // CAD models are usually Z-up
    this.camera.position.set(1, -1.3, 0.9);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.15;
    this.controls.addEventListener("change", () => this.invalidate());

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x5a5a5a, 1.4));
    const head = new THREE.DirectionalLight(0xffffff, 1.9);
    head.position.set(0.3, 0.6, 1);
    this.camera.add(head);
    this.scene.add(this.camera);
    this.scene.add(this.surfaceGroup);

    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    let down: { x: number; y: number } | null = null;
    this.renderer.domElement.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 4) this.pick(e);
      down = null;
    });
    const tick = () => {
      if (this.controls.update()) this.needsRender = true;
      if (this.needsRender) {
        this.needsRender = false;
        this.renderer.render(this.scene, this.camera);
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  invalidate(): void { this.needsRender = true; }

  private resize(): void {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + "px";
    this.renderer.domElement.style.height = h + "px";
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  // ------------------------------------------------------------- content
  /** Level-0 preview (the fitted overall shape): translucent fill plus edges. */
  setOverlay(positions: Float32Array, indices: Uint32Array): void {
    this.clearOverlay();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setIndex(new THREE.BufferAttribute(indices, 1));
    const grp = new THREE.Group();
    grp.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({
      color: 0x3987e5, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide })));
    grp.add(new THREE.LineSegments(new THREE.EdgesGeometry(g),
      new THREE.LineBasicMaterial({ color: 0x3987e5, transparent: true, opacity: 0.9 })));
    this.overlay = grp;
    this.scene.add(grp);
    this.invalidate();
  }

  clearOverlay(): void {
    if (!this.overlay) return;
    this.scene.remove(this.overlay);
    this.overlay.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      (m.material as THREE.Material | undefined)?.dispose?.();
    });
    this.overlay = null;
    this.invalidate();
  }

  /** Colour the mesh by one refinement level's grouping (directions, planes, ...). */
  showFaceLevel(labels: Uint32Array): void {
    if (!this.meshObj) return;
    this.levelView = true;
    const cat = categorical();
    const base = linear([0.72, 0.71, 0.68]);
    let max = 0;
    for (let i = 0; i < labels.length; i++) if (labels[i] > max) max = labels[i];
    const cache = new Map<number, THREE.Color>();
    const attr = this.meshObj.geometry.getAttribute("color") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    for (let f = 0; f < labels.length; f++) {
      const l = labels[f];
      let c = cache.get(l);
      if (!c) {
        // the last label of a level means "not explained yet": keep it dark neutral
        c = l === max ? linear([0.3, 0.3, 0.29]) : linear(hexRGB(cat[(l * 5) % 8])).lerp(base, 0.15);
        cache.set(l, c);
      }
      for (let k = 0; k < 3; k++) {
        const o = f * 9 + k * 3;
        arr[o] = c.r; arr[o + 1] = c.g; arr[o + 2] = c.b;
      }
    }
    attr.needsUpdate = true;
    this.applyLayers();
  }

  endLevelView(): void {
    this.levelView = false;
    this.clearOverlay();
    this.applyLayers();
  }

  clearAll(): void {
    for (const o of [this.meshObj, this.pointsObj]) {
      if (o) { this.scene.remove(o); o.geometry.dispose(); (o.material as THREE.Material).dispose(); }
    }
    this.meshObj = null;
    this.pointsObj = null;
    this.clearSurfaces();
    this.clearOverlay();
    this.levelView = false;
    this.labels = this.faceLabels = this.normals = this.pointRes = null;
    this.regionIdx.clear();
    this.regionRGB.clear();
    this.highlight = null;
    this.invalidate();
  }

  setMesh(pos: Float32Array, idx: Uint32Array): void {
    this.clearAll();
    const nf = idx.length / 3;
    const P = new Float32Array(nf * 9);
    for (let f = 0; f < nf; f++) {
      for (let k = 0; k < 3; k++) {
        const v = idx[f * 3 + k] * 3;
        const o = f * 9 + k * 3;
        P[o] = pos[v]; P[o + 1] = pos[v + 1]; P[o + 2] = pos[v + 2];
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(P, 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(P.length), 3));
    g.computeVertexNormals();
    g.computeBoundingBox();
    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.75, metalness: 0.05, side: THREE.DoubleSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, transparent: true,
    });
    this.meshObj = new THREE.Mesh(g, mat);
    this.scene.add(this.meshObj);
    this.diag = g.boundingBox!.getSize(new THREE.Vector3()).length() || 1;
    this.recolorMesh();
    this.applyLayers();
    this.fitView();
  }

  setPoints(pos: Float32Array, nrm: Float32Array): void {
    if (this.pointsObj) {
      this.scene.remove(this.pointsObj);
      this.pointsObj.geometry.dispose();
    }
    this.clearSurfaces();
    this.labels = this.pointRes = null;
    this.regionIdx.clear();
    this.normals = nrm;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(pos.length), 3));
    const n = pos.length / 3;
    const size = this.diag * Math.min(0.012, Math.max(0.0025, 0.9 / Math.sqrt(n)));
    const mat = new THREE.PointsMaterial({
      size, sizeAttenuation: true, vertexColors: true, map: this.disc, alphaTest: 0.5, transparent: false,
    });
    this.pointsObj = new THREE.Points(g, mat);
    this.scene.add(this.pointsObj);
    this.recolorPoints();
    this.applyLayers();
  }

  setLabels(labels: Uint32Array, faceLabels: Uint32Array, slots: Map<number, number>): void {
    this.labels = labels;
    this.faceLabels = faceLabels;
    this.clearSurfaces();
    const counts = new Map<number, number>();
    for (const l of labels) counts.set(l, (counts.get(l) ?? 0) + 1);
    const fill = new Map<number, number>();
    this.regionIdx.clear();
    for (const [rid, c] of counts) { this.regionIdx.set(rid, new Uint32Array(c)); fill.set(rid, 0); }
    labels.forEach((l, i) => { const k = fill.get(l)!; this.regionIdx.get(l)![k] = i; fill.set(l, k + 1); });
    this.pointRes = new Float32Array(labels.length).fill(NaN);
    this.setSlots(slots);
  }

  setSlots(slots: Map<number, number>): void {
    const cat = categorical();
    this.regionRGB.clear();
    for (const [rid, s] of slots) this.regionRGB.set(rid, hexRGB(cat[s]));
    for (const [rid, m] of this.surfaces) (m.material as THREE.MeshStandardMaterial).color = linear(this.rgbOf(rid));
    this.recolorPoints();
    this.recolorMesh();
  }

  private rgbOf(rid: number): RGB { return this.regionRGB.get(rid) ?? NEUTRAL; }

  setResiduals(rid: number, values: Float32Array): void {
    const idx = this.regionIdx.get(rid);
    if (!idx || !this.pointRes || idx.length !== values.length) return;
    for (let k = 0; k < idx.length; k++) this.pointRes[idx[k]] = values[k];
    if (this.colorMode === "residual") this.recolorPoints(idx);
  }

  clearResiduals(rid: number): void {
    const idx = this.regionIdx.get(rid);
    if (!idx || !this.pointRes) return;
    for (const i of idx) this.pointRes[i] = NaN;
    if (this.colorMode === "residual") this.recolorPoints(idx);
  }

  setSurface(rid: number, positions: Float32Array, indices: Uint32Array): void {
    let m = this.surfaces.get(rid);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    g.setIndex(new THREE.BufferAttribute(indices, 1));
    g.computeVertexNormals();
    if (!m) {
      const mat = new THREE.MeshStandardMaterial({
        color: linear(this.rgbOf(rid)), roughness: 0.45, metalness: 0.1, side: THREE.DoubleSide,
        transparent: true, opacity: 1, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
      });
      m = new THREE.Mesh(g, mat);
      m.userData.rid = rid;
      this.surfaces.set(rid, m);
      this.surfaceGroup.add(m);
    } else {
      m.geometry.dispose();
      m.geometry = g;
    }
    this.applyHighlight();
    this.invalidate();
  }

  removeSurface(rid: number): void {
    const m = this.surfaces.get(rid);
    if (!m) return;
    this.surfaceGroup.remove(m);
    m.geometry.dispose();
    (m.material as THREE.Material).dispose();
    this.surfaces.delete(rid);
    this.invalidate();
  }

  clearSurfaces(): void {
    for (const rid of [...this.surfaces.keys()]) this.removeSurface(rid);
  }

  // ------------------------------------------------------------- display
  setLayer(name: "mesh" | "points" | "surfaces", on: boolean): void {
    this.layers[name] = on;
    this.applyLayers();
  }

  private applyLayers(): void {
    const hasPts = !!this.pointsObj && this.layers.points && !this.levelView;
    const hasSurf = this.surfaces.size > 0 && this.layers.surfaces;
    if (this.meshObj) {
      this.meshObj.visible = this.layers.mesh;
      const mat = this.meshObj.material as THREE.MeshStandardMaterial;
      mat.opacity = hasPts || hasSurf ? 0.22 : 1;
      mat.depthWrite = !(hasPts || hasSurf);
    }
    if (this.pointsObj) this.pointsObj.visible = this.layers.points && !this.levelView;
    this.surfaceGroup.visible = this.layers.surfaces;
    this.invalidate();
  }

  setColorMode(m: ColorMode): void {
    this.colorMode = m;
    this.recolorPoints();
  }

  setHighlight(rid: number | null): void {
    this.highlight = rid;
    this.recolorPoints();
    this.applyHighlight();
  }

  private applyHighlight(): void {
    for (const [rid, m] of this.surfaces) {
      const mat = m.material as THREE.MeshStandardMaterial;
      const dim = this.highlight !== null && rid !== this.highlight;
      mat.opacity = dim ? 0.18 : 1;
      mat.depthWrite = !dim;
    }
    this.invalidate();
  }

  private recolorMesh(): void {
    if (!this.meshObj) return;
    const attr = this.meshObj.geometry.getAttribute("color") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const nf = arr.length / 9;
    const base = linear([0.72, 0.71, 0.68]);
    const cache = new Map<number, THREE.Color>(); // one colour per region, not per triangle
    for (let f = 0; f < nf; f++) {
      let c = base;
      if (this.faceLabels) {
        const l = this.faceLabels[f];
        c = cache.get(l) ?? linear(this.rgbOf(l)).lerp(base, 0.35);
        cache.set(l, c);
      }
      for (let k = 0; k < 3; k++) {
        const o = f * 9 + k * 3;
        arr[o] = c.r; arr[o + 1] = c.g; arr[o + 2] = c.b;
      }
    }
    attr.needsUpdate = true;
    this.invalidate();
  }

  private recolorPoints(subset?: Uint32Array): void {
    if (!this.pointsObj) return;
    const attr = this.pointsObj.geometry.getAttribute("color") as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const n = arr.length / 3;
    const tmp = new THREE.Color();
    const lin = new Map<number, THREE.Color>();
    for (const [rid, rgb] of this.regionRGB) lin.set(rid, linear(rgb));
    const neutral = linear(NEUTRAL);
    const dimTo = linear([0.35, 0.35, 0.34]);
    const scale = this.residualScale || 1;
    const one = (i: number) => {
      const l = this.labels ? this.labels[i] : -1;
      if (this.colorMode === "normal" && this.normals) {
        tmp.setRGB(this.normals[i * 3] * 0.5 + 0.5, this.normals[i * 3 + 1] * 0.5 + 0.5,
          this.normals[i * 3 + 2] * 0.5 + 0.5, THREE.SRGBColorSpace);
      } else if (this.colorMode === "residual") {
        const r = this.pointRes ? this.pointRes[i] : NaN;
        if (Number.isFinite(r)) tmp.setRGB(...diverging(r / scale), THREE.SRGBColorSpace);
        else tmp.copy(neutral);
      } else {
        tmp.copy(l >= 0 ? (lin.get(l) ?? neutral) : neutral);
      }
      if (this.highlight !== null && l !== this.highlight) tmp.lerp(dimTo, 0.75);
      arr[i * 3] = tmp.r; arr[i * 3 + 1] = tmp.g; arr[i * 3 + 2] = tmp.b;
    };
    if (subset) for (const i of subset) one(i);
    else for (let i = 0; i < n; i++) one(i);
    attr.needsUpdate = true;
    this.invalidate();
  }

  fitView(): void {
    const box = new THREE.Box3();
    if (this.meshObj) box.expandByObject(this.meshObj);
    else if (this.pointsObj) box.expandByObject(this.pointsObj);
    if (box.isEmpty()) return;
    const c = box.getCenter(new THREE.Vector3());
    const r = box.getSize(new THREE.Vector3()).length() / 2;
    const vHalf = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const hHalf = Math.atan(Math.tan(vHalf) * this.camera.aspect);
    const dist = (r / Math.sin(Math.min(vHalf, hHalf))) * 1.02;
    const dir = new THREE.Vector3(1, -1.3, 0.9).normalize();
    this.camera.position.copy(c).addScaledVector(dir, dist);
    this.camera.near = Math.max(dist / 1000, 1e-4);
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(c);
    this.controls.update();
    this.invalidate();
  }

  private pick(e: PointerEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    let rid: number | null = null;
    if (this.layers.surfaces && this.surfaces.size) {
      const hit = ray.intersectObjects([...this.surfaces.values()], false)[0];
      if (hit) rid = hit.object.userData.rid as number;
    }
    if (rid === null && this.meshObj && this.faceLabels) {
      const hit = ray.intersectObject(this.meshObj, false)[0];
      if (hit && hit.faceIndex != null) rid = this.faceLabels[hit.faceIndex];
    }
    this.onPick?.(rid);
  }
}
