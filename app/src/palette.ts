// Colour roles (reference data-viz palette, validated light + dark).
const CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
const CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];

export const isDark = () =>
  document.documentElement.dataset.theme === "dark" ||
  (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);

export const categorical = () => (isDark() ? CAT_DARK : CAT_LIGHT);

/**
 * Region colours. A model can have dozens of regions but there are only eight
 * categorical hues, so hues are assigned by greedy graph colouring: regions that
 * touch never share a hue. Identity comes from the region list (id + swatch).
 */
export function regionSlots(ids: number[], adjacency: [number, number][]): Map<number, number> {
  const nb = new Map<number, Set<number>>();
  for (const id of ids) nb.set(id, new Set());
  for (const [a, b] of adjacency) {
    nb.get(a)?.add(b);
    nb.get(b)?.add(a);
  }
  const slot = new Map<number, number>();
  const uses = new Array(8).fill(0);
  for (const id of ids) {
    const taken = new Set([...nb.get(id)!].map((n) => slot.get(n)).filter((s) => s !== undefined));
    // least-used hue among those no neighbour has (spreads all eight hues)
    let pick = -1;
    for (let s = 0; s < 8; s++) if (!taken.has(s) && (pick < 0 || uses[s] < uses[pick])) pick = s;
    if (pick < 0) pick = uses.indexOf(Math.min(...uses));
    slot.set(id, pick);
    uses[pick]++;
  }
  return slot;
}

// Diverging blue <-> neutral <-> red for signed fit error (neutral = on the surface).
const NEG = [0x2a / 255, 0x78 / 255, 0xd6 / 255];
const MID = [0x9a / 255, 0x99 / 255, 0x94 / 255];
const POS = [0xe3 / 255, 0x49 / 255, 0x48 / 255];

/** t in [-1, 1] -> sRGB triple (0..1). */
export function diverging(t: number): [number, number, number] {
  const c = Math.max(-1, Math.min(1, t));
  const end = c < 0 ? NEG : POS;
  const k = Math.abs(c);
  return [MID[0] + (end[0] - MID[0]) * k, MID[1] + (end[1] - MID[1]) * k, MID[2] + (end[2] - MID[2]) * k];
}

export const NEUTRAL: [number, number, number] = [0.55, 0.55, 0.53];
