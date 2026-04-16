# JanusMesh: Fast, Training-Free Text-Driven 3D Visual Illusions

![Teaser](docs/teaser.png)

*JanusMesh is a training-free pipeline that turns two text prompts into a single 3D asset whose semantics read differently from different viewpoints—typically in a few minutes—by fusing dual-branch 3D generation in voxel/SDF space and refining appearance with view-conditioned 2D diffusion on the mesh.*

## Abstract

Creating 3D visual illusions—a single 3D mesh that reveals entirely different semantics from various viewing angles—is a fascinating but tough challenge. Existing optimization-based methods are slow and can produce oversaturated colors. In contrast, naive stitching approaches fail to produce geometrically coherent objects. This results in visible unnatural seams and semantic leaks.

In this paper, we present a fast and training-free framework for generating text-driven 3D visual illusions. Our approach decouples the generation into two stages. First, we propose a cross-space dual-branch denoising process. This process dynamically decodes 3D latents into voxel space for CLIP-guided orientation alignment and Signed Distance Field (SDF) blending, which ensures seamless geometric fusion. Second, we introduce a view-conditioned texture synthesis module that projects and aggregates view-specific 2D diffusion priors onto the fused geometry.

Extensive experiments demonstrate that our method generates highly realistic, dual-semantic 3D illusions in just 3–5 minutes. It significantly outperforms existing methods in geometric integrity, semantic recognizability, and efficiency.

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-4285F4?logo=adobeacrobatreader&logoColor=white)](docs/JanusMesh_paper.pdf)
[![Project website](https://img.shields.io/badge/Project-Website-orange)](#)

---

## Repository layout

| Path | Role |
|------|------|
| `example_text.py` | Main entry: dual-prompt TRELLIS run and optional SyncTweedies mesh texturing |
| `trellis/` | Modified TRELLIS library (text-to-3D pipeline, samplers, renderers) |
| `dataset_toolkits/` | Used by voxel blending (`_render` in samplers) |
| `configs/` | Optional local configs |
| `clip/` | CLIP view scoring and rotation heuristics for `example_text.py` |
| `SyncTweedies/` | Mesh texturing app (`--app mesh`); see `SyncTweedies/README.md` |

Excluded on purpose from this bundle: legacy `evaluate/`, `clip2/`, batch scripts, `outputs/`, and large `SyncTweedies/data/` (not required for `--app mesh` with a user-supplied GLB).

## Environment (single env: `janusmesh`)

**Goal:** one conda env for both Trellis and SyncTweedies.

1. On your machine (where `trellis` and `synctweedies` already work), run:
   ```bash
   bash scripts/export_env_snapshots.sh trellis synctweedies
   ```
2. Follow **`docs/ENVIRONMENT.md`** to merge exports into a root **`environment.yml`** named `janusmesh`, test `conda env create -f environment.yml`, then commit that file.
3. Set `run_synctweedies_mesh(..., conda_env_name="janusmesh")` in `example_text.py` (or keep two envs until the merged file is verified).

**Until `environment.yml` is published**, you can still use two envs as before and rely on `SyncTweedies/environment.yml` for the texturing stack.

## Run

From this directory:

```bash
cd JanusMesh
python example_text.py --prompt1 "a frog sitting on a leaf" --prompt2 "a turtle" --case 2 --manual_rotation_step 0
```

- **`--case` 1 or 2:** generation only (fixed voxel split). For case 2, set **`--manual_rotation_step`** (90° steps) as needed.
- **`--case` 3:** CLIP pose search (step 1) then generation (step 2).
- Weights load from Hugging Face (`microsoft/TRELLIS-text-xlarge`); ensure network/HF access or a local cache.

## Secrets

- **`OPENAI_API_KEY`**: Only needed if you call GPT-based helpers in `clip/render_dual_eval.py`. `example_text.py` uses CLIP when **`--case 3`** (step 1 before generation) and does not need OpenAI for that path; the client is created lazily on first GPT use.

## Notes

- SyncTweedies path is **`JanusMesh/SyncTweedies`** (sibling of `example_text.py`), not the parent of the repo.
- If you previously had an API key committed in upstream `TRELLIS/clip/render_dual_eval.py`, **rotate that key** in the OpenAI dashboard; this release uses environment variables only.
