# Held-back segmenter changes

Nothing is held back at the moment.

- `_thin_chains` (pipe bends sliced into cylinder bands fitted as one torus) was held
  back here until pipe-elbow STEP export worked. It went into
  `backend/meshtocad/meshseg.py` in round 5 (2026-10-28), together with the elbow export
  fix in `cad.py` (tori that wrap their tube are written as TOROIDAL_SURFACE with a seam
  along the pipe). The old patch script is kept outside the repo in
  `.cache/patch_thin_chains_merged.py`.
