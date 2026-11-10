# Assets — provenance and usage

All images in this folder are AI-generated artwork made for this project
(the custom renderer plan, 2026-10-28). They are original generations, not
third-party stock or screenshots — no external licence applies to the files
themselves. If you reuse them outside this project, check the terms of the
tool that generated them.

## Files used by the renderer (`src/ll/renderer.py`)

| File | What it is | Used as |
|---|---|---|
| `space_backdrop.png` | Starfield backdrop | Full-screen background, drawn behind the scene |
| `lander.png` | Lander sprite (transparent) | The lander, drawn from the live env state |
| `engine_plume.png` | Engine plume sprite | Plume under the nozzle while the main engine fires |
| `Code_Generated_Image.png` | Engine-core sprite | Bright core inside the plume (main engine only) |
| `Gemini_Generated_Image_cpnib4cpnib4cpni.jpg` | Lunar regolith texture | Terrain texture (tried first) |
| `lunar_regolith.png` | Lunar regolith texture | Terrain texture (fallback if the JPG above is missing) |

The renderer falls back to procedural drawing if an asset file is missing,
so a deleted texture never breaks the viewer — it just looks plainer.

## Removed

Two further Gemini-generated images
(`Gemini_Generated_Image_47v9vs47v9vs47v9.jpg`,
`Gemini_Generated_Image_63117u63117u6311.jpg`, ~5 MB) were deleted on
2026-11-10: nothing in the codebase referenced them.
