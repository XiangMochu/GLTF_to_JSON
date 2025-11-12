# gltfextract - glTF/GLB → baked JSON (skinned + animated)

**One‑file extractor that bakes skinned glTF/GLB into integer‑quantized JSON frames** (keeps vertex order, stitches clips, samples colors, and includes skeleton + weights). Ships with a tiny Tk GUI and a CLI.

---

## Quickstart

```bash
# CLI
python gltfextract.py --in path/to/model.glb --out model_baked.json --fps 20 --target 2048 --pretty

# GUI (just run with no args)
python gltfextract.py
```

* `--in/-i`  input .glb/.gltf
* `--out/-o` output .json
* `--fps`    sampling rate for animation baking (default 20)
* `--target` target integer **height** used for quantization (default 2048)
* `--pretty` pretty‑print JSON


---

## Features

* **Correct hierarchical skinning.** Per‑frame `jointMatrices = worldAnimated(joint) * inverseBindMatrix[j]`, then CPU‑reskins vertices and optionally applies the mesh node’s world transform.
* **Vertex order preserved.** No reindexing - indices map straight to your runtime.
* **Clip stitching + metadata.** Concatenates animation channels into a global timeline and exports `animationClips` with `startFrame/endFrame/durationSeconds/fps`.
* **Color sampling.** Uses vertex colors if present; otherwise samples PBR base‑color (factor/texture), falling back to a neutral color when textures are unavailable. Optional Pillow support for texture reads.
* **Output tailored for game runtimes.** Integer‑quantized per‑frame arrays `vx/vy/vz`, face lists, and legacy `baseX/framesX` style arrays for drop‑in use.
* **Skin block.** Emits joints (with parent links, inverseBind, bindLocal TRS), per‑frame `jointMatrices`, and per‑vertex `JOINTS_0/WEIGHTS_0` (normalized) when the source mesh is skinned.
* **Auto‑mesh pick.** Chooses the densest mesh by face count.
* **Tiny GUI.** Tkinter UI for quick exports; same codepath as CLI.

---

## Installation

* Python 3.9+ (tested on 3.10+)
* Optional: **Pillow** (`pip install pillow`) for texture sampling in color baking
* Tkinter is included with most Python builds (used for the GUI)

No packaging required, it’s a single file. Add to your project or run it standalone.

---

## Output JSON (schema sketch)

```jsonc
{
  "frames": [
    { "vx": [int...], "vy": [int...], "vz": [int...] },
    // ... one entry per baked frame
  ],
  "faces": { "a": [int...], "b": [int...], "c": [int...] },
  "vcol":  { "r": [0..255], "g": [0..255], "b": [0..255] },

  // Convenience legacy arrays (first frame + all frames split per axis)
  "baseX": [int...], "baseY": [int...], "baseZ": [int...],
  "framesX": [[int...], ...],
  "framesY": [[int...], ...],
  "framesZ": [[int...], ...],

  "animationClips": [
    { "name": "clip", "startFrame": 0, "endFrame": 24, "frameCount": 25,
      "durationSeconds": 1.25, "fps": 20 }
  ],

  "fps": 20,
  "qscale": 2048.0, // scale used to integer‑quantize positions

  // Present only for skinned meshes
  "skin": {
    "joints": [
      {
        "index": 0,
        "node": 7,
        "name": "UpperArm.R",
        "parent": -1,
        "inverseBindMatrix": [16 floats],
        "bindLocal": { "translation": [x,y,z], "rotation": [w,x,y,z], "scale": [x,y,z] }
      }
    ],
    "jointNodes": [7, 8, 9, ...],
    "jointMatrices": [ [16 floats], ... per frame ... ],
    "weights": {
      "joints": [[j0,j1,j2,j3], ... per vertex ...],
      "weights": [[w0,w1,w2,w3], ...] // normalized
    }
  }
}
```

### How quantization works

* The script computes the source mesh’s **Y‑extent** and sets `qscale = target_int_height / height` (default 2048).
* Positions are multiplied by `qscale` and rounded to `int` per frame.

> If you prefer different axis/range logic, change `target_int_h` via CLI, or adapt the code.

---

## Programmatic use

```python
from pathlib import Path
from gltfextract import export

# bake to JSON dict (also writes to disk)
export(Path("model.glb"), Path("model_baked.json"), sample_fps=20, target_int_h=2048, pretty=True)
```

---

## Limitations / notes

* Interpolation uses simple lerp for quaternions (good enough for many cases; change to slerp if needed).
* Picks the **densest** mesh; multi‑mesh scenes are not all exported yet.
* Morph targets are not baked (PRs welcome).
* Texture sampling for colors requires Pillow; otherwise a base color is used.
* Output arrays are intentionally **integer** for compact, cache‑friendly use at runtime.

---

## License

MIT

---
