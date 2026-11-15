# Offline CAD tooling

This research command reads a player's existing local voxel cache. It never downloads files,
starts a VM, edits source caches, installs a model into the renderer, or changes live rendering.
Run from the repository root with the configured pnpm version and installed dependencies.
Board rendering additionally requires Python with NumPy and Pillow.

```bash
pnpm exec tsx scripts/remaster/cad/run.ts \
  --id "$MODEL_ID" --cache-root "$SOURCE_CACHE" \
  --palette "$UNIT_PALETTE_PNG" --house neutral --out "$OUTPUT_DIRECTORY"
```

Every resource path is explicit. Output must resolve outside the checkout and must not contain
source inputs or symbolic links. Use a fresh output directory for each experiment. Narrow the
cache root if several distinct source caches contain the requested ID. Discovery searches at
most four nested directories, accepts voxel-section caches, and excludes generated
`vehicles-hd` projections. It includes every section and exports section bounds, HVA frame 0,
raw occupancy, the filled solid, source paint/remap, and current `buildStandInPartGeometry`
vertices. Its lattice-to-world affine must agree with those vertices within `1e-5` world units.

`--house neutral` uses sRGB `(0.6, 0.6, 0.6)`. Alternatively provide `--house 0.1,0.3,0.8`.
These are caller-selected colours, not a claim about a faction's default player colour. Remap
membership comes exactly from source palette indices 16 through 31; source ramp position
controls shade while the parameter controls hue. CAD paint queries the nearest original
exposed source face in its normal direction, so it never invents geometric stripes. The
default hard-surface look is neutral; `ra2HardSurfaceLook` constructs an isolated player policy.

The generic modules in `packages/hd-renderer/src/` own:

- `hardSurfacePropose.ts`: part-local mirror evidence, bounded broad BSP regions, snapped
  axis/1:1/1:2/2:1 planes plus configurable ratios, occupancy-supported circular hypotheses,
  and disconnected parallel tube lanes with shared centres on axial refinements. Semantic
  names are geometric hypotheses, not recognition of a particular game's model.
- `hardSurfaceSilhouette.ts`: source surface extraction, deterministic two-sided triangle
  rasterization at pixel centres, clipping rejection, and per-view overlap/residual counts.
  Camera and section placement are injected. Default research masks use 900 by 900 pixels,
  scale 8, and the established isometric projection. This pixel-centre convention differs
  from the old OpenCV inclusive-edge masks; do not compare rounded historical IoUs as if
  they came from identical rasterizers.
- `hardSurfaceRefine.ts`: worst-facing residual localization, source-region splitting and
  sloped-plane/radius adjustment, with formal `hardSurfaceMetrics` and every-view silhouette
  selection. It fixes partition planes to avoid opening cracks and never invokes an exact
  or per-voxel cuboid cover. Once both surface gates pass, silhouette search cannot reopen
  them. Non-finite formal values remain visible and cannot satisfy the corresponding gate.
- `hardSurfaceRecess.ts`: broad connected overfill pockets with source occupancy on opposite
  sides. Convex subtraction candidates need at least twelve source-empty cells, a long span
  of six cells, and at most one percent source overlap. Tiny dimples, paint-only marks and
  unconstrained exterior space cannot become residual-cell cuts. Every subtraction still
  goes through the same expanded primitive, surface and silhouette gates.

The RA2 composition script maps chassis/crown/tube policy and applies expanded primitive
budgets 130/35/3. A multi-section part divides its budget between sections. Lateral reflection
is reported; source asymmetries are retained rather than silently replaced by mirrored geometry.
Round hypotheses need source occupancy support; labels alone never authorize added wheels,
drums, panels, or other decoration. There is no learned semantic model or promise that every
source feature will be recognized. Broad convex partitions can still underfit concavities or
miss thin source features. Numerical, visual and runtime acceptance remain separate gates.

`--evaluations 32` and `--seconds 300` bound refinement **per section**. Proposal/export time
is reported separately. The last in-flight formal evaluation completes before a time-budget
stop, so wall time can exceed the soft compute budget. Failure, budget exhaustion and a
search with no improving broad candidate are explicit outcomes, not passes. Every evaluated
candidate and the last selected spec are checkpointed synchronously; crashes/timeouts retain
those files. `--no-render` omits PNGs for diagnostics, and `--export-only` stops after exports.
`--python` selects the interpreter. `--font` selects a CJK font for Chinese labels; without it,
boards use portable English text. `--cell` can be passed directly to `render.py` if a larger
tile is required; clipping fails instead of silently rescaling the model.

Deliveries contain source hashes, exports, proposals, the authoritative CAD mesh, formal
metrics, every facing's IoU, 1x/nearest-neighbour 2x boards, a turntable, automatic timings,
draft reports and `manifest.json`. Reports must separately account for tool development,
manual diagnosis/selection and visual review. Empty model-coordinate decisions do not prove
zero hand-directed labour. After reviewing or annotating a delivery, refresh its hashes:

```bash
pnpm exec tsx scripts/remaster/cad/manifest.ts "$OUTPUT_DIRECTORY"
pnpm exec vitest run packages/hd-renderer/tests/hardSurface
pnpm run check
```

The primitive budgets, bidirectional p99 ceiling of 0.85, and each part/assembly's eight IoU
floors of 0.97 are independent. A reviewer must open both board scales to check continuous
plates, source-supported cylinders, unchanged proportions and no additions. No offline
artifact establishes runtime behaviour, GPU appearance, performance or artistic approval.
