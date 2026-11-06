# Regression tests

One command checks a backend change before it goes into the app. Run from the project root
with the app's Python (`runtime/python/python.exe`). The STEP suite also needs the dev tools
in `.cache/devtools` (OpenCASCADE) to open the exported files like a CAD tool would.

```
python scripts/regress/regress.py seg  --backend .cache/rNN/be            # 613 models, ~25 min
python scripts/regress/regress.py seg  --backend .cache/rNN/be --list quick  # 72 models, ~4 min
python scripts/regress/regress.py step --backend .cache/rNN/be            # 41 models, ~40 min
python scripts/regress/regress.py selftest --backend .cache/rNN/be
```

Each run lands in `.cache/regress/<tag>` (`--tag`, default a timestamp) and is compared with
the saved baseline in `baselines/`. The exit code is 1 when something got worse.

- **seg** segments every model with known true CAD faces (193 real parts + 420 synthetic) and
  scores it: `incomplete` (true faces left in pieces), `impure` (faces spilling onto other
  surfaces), `wrong_type`, `extra` (one true surface split into several faces of ours),
  `mixed`, `jagged` (saw-teeth on borders). Lower is better. `--labels` keeps every model's
  faces for `census.py`. The stored meshes come from an older loader, so the worker repeats the
  backend's load repair (`cloud.split_nonmanifold`) and finds the bodies again when it changes a mesh.
- **step** runs the app's own pipeline (load, segment, export STEP; `pipeline.py`) and opens
  the file with OpenCASCADE: failing faces, open shells, solids, missing area against the
  source. List: `step_quick.txt` (user parts, real models with problems, a fixed clean
  sample, synthetic parts with problems). A model that times out or exits without a result
  (often just a busy machine) gets one more run at the end before it counts as failed.
- **census.py RUN_TAG [--real|--synthetic]** lists the extra faces of a `--labels` run by
  kind (true type, our type, speck / sliver / tile, whether the piece touches the face that
  mainly covers its true face), with each kind's share of the score. Pieces "apart" from
  that face cannot be merged into it; on real parts most are copies of an instanced part.
- `compare seg|step TAG_A TAG_B` compares two runs; `baseline seg|step TAG` makes a run the
  new baseline (do this after porting a change into `backend/`).

Known scorer limit: a part placed twice in an assembly repeats its true-face numbers, so its
two copies of a face count as one face split in two ("tiles apart"); those are correct.
