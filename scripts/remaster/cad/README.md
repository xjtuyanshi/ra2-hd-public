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
The affine is derived from the shared `ra2/voxelSectionFrame.ts` policy, which also places live
voxel, smooth and CAD sections; the CLI does not maintain a second HVA/axis-swap implementation.

`--house neutral` uses sRGB `(0.6, 0.6, 0.6)`. Alternatively provide `--house 0.1,0.3,0.8`.
These are caller-selected colours, not a claim about a faction's default player colour. Remap
membership comes exactly from source palette indices 16 through 31; source ramp position
controls shade while the parameter controls hue. CAD paint queries the nearest original
exposed source face in its normal direction, so it never invents geometric stripes. The
default hard-surface look is neutral; `ra2HardSurfaceLook` constructs an isolated player policy.

The generic modules in `packages/hd-renderer/src/` own:

- `hardSurfacePropose.ts`: part-local mirror evidence, bounded broad BSP regions, snapped
  axis/1:1/1:2/2:1 planes plus configurable ratios, occupancy-supported circular hypotheses,
  and parallel tube lanes with source-step dynamic-programming chains. A majority of disjoint
  axial slices can separate tubes joined at a short root. Axial gaps are never bridged.
  Each chain shares one axis and receives its own caller-supplied segment budget. A real
  one-slice terminal muzzle/cap is eligible; interior segments still require axial extent. Slice-median
  axes prevent a short asymmetric base from displacing a long tube; centre search moves every
  segment of one coaxial group together. Semantic
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
  them. Once all eight silhouette gates close, surface search cannot reopen those either.
  Up to four source-supported broad split alternatives can be evaluated per region;
  a rejected occupancy-best cut does not exhaust its entire region. Crown regions begin at
  the detected bilateral plane when source support is strong. Only exactly reflected source
  regions are coupled, including their splits and plate adjustments; asymmetric source
  details remain independent. Non-finite formal values remain visible and cannot satisfy the corresponding gate.
  Parameter-candidate memoization includes the current primitive or entire coaxial group, so
  changing a radius cannot suppress a newly useful centre move or another step in the same direction.
- `hardSurfaceRecess.ts`: broad connected overfill pockets with source occupancy on opposite
  sides. Convex subtraction candidates need at least twelve source-empty cells, a long span
  of six cells, and at most one percent source overlap. Broad source-face trim panels are also eligible when their thickness is at most two cells
  and at least half their cells directly contact an occupied source face. Tiny dimples,
  paint-only marks and unconstrained exterior space cannot become residual-cell cuts. Every subtraction still
  goes through the same expanded primitive, surface and silhouette gates. If a connected
  pocket wraps around source material, bounded broad subdivisions can yield safe convex
  sub-pockets. Recursion stops at depth three; the same extent/enclosure/source-overlap
  requirements apply to every child.

The RA2 composition script maps chassis/crown/tube policy and applies expanded primitive
budgets body <=130, turret <=35, and <=3 primitives per detected barrel tube. A multi-section
body or turret divides its budget between sections. Barrel lanes receive independent budgets
(the twin-tube case permits six total). Lateral reflection
is reported; source asymmetries are retained rather than silently replaced by mirrored geometry.
Round hypotheses need source occupancy support; labels alone never authorize added wheels,
drums, panels, or other decoration. There is no learned semantic model or promise that every
source feature will be recognized. Broad convex partitions can still underfit concavities or
miss thin source features. Numerical, visual and runtime acceptance remain separate gates.

`--evaluations 32` and `--seconds 300` bound refinement **per section per search trial**. Proposal/export time
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
draft reports and `manifest.json`. `source_snapshot.json` records the starting revision,
working-tree status and tooling hashes; the manifest also reports source files changed during
a run, so a dirty or concurrently edited checkout cannot masquerade as an immutable revision. Reports must separately account for tool development,
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

## Residual evidence

The diagnostic below consumes an existing offline delivery; it never changes its spec or the
formal acceptance metric. It writes projected surface-sample heat maps at the same native
research scale and at nearest-neighbour 4x. Dark pixels are unsampled; this is a depth-selected
sample board, not a silhouette mask or a runtime capture.

```bash
pnpm exec tsx scripts/remaster/cad/residual.ts --input "$DELIVERY" --out "$RESIDUAL_OUTPUT"
python3 scripts/remaster/cad/residual_board.py --input "$DELIVERY" --residual "$RESIDUAL_OUTPUT"
```

`residual_summary.json` compares unchanged formal NN distances with independent triangle
surface distances, sampling seed 11, a zero-chamfer geometry probe, and half-voxel translation
probes. Translation probes report their stratified subset sizes and are never acceptance.
The triangle query in `hardSurfaceResidual.ts` uses a tested AABB hierarchy. CAD-to-voxel
queries use the exposed source surface. Voxel-to-CAD queries include all input triangles,
including buried primitive faces, and are therefore explicitly a lower bound on distance to
the actual union boundary. A lower bound above the gate proves a real geometry discrepancy;
a lower bound below it does not prove a sampling artefact. None of these diagnostics replaces
`hardSurfaceMetrics`, changes thresholds, or makes a failed fit pass.

## Batch execution

```bash
pnpm exec tsx scripts/remaster/cad/batch.ts \
  --ids "$MODEL_IDS" --cache-root "$SOURCE_CACHE" --palette "$UNIT_PALETTE_PNG" \
  --house neutral --out "$BATCH_OUTPUT" --concurrency 4 \
  --evaluations 1000 --seconds 1200 --cell 256 --pre-gate
pnpm exec tsx scripts/remaster/cad/summary.ts "$BATCH_OUTPUT"
```

`MODEL_IDS` is an explicit comma-separated list, without duplicates. The batch writes each
vehicle under `units/`, continues after child failures, and retains actual exit codes and
elapsed minutes in `batch_status.json`. `--pre-gate` labels exploratory results before the
primary vehicles satisfy full acceptance. A gated label alone never constitutes acceptance:
`summary.ts` requires successful process and metric evidence plus an independently written
`visual_review.json` with `pass`, `reason`, and `agentDirectedMinutes`. Missing reviews stay
UNVERIFIED. Record actual review windows and all shared direction separately in HOURS.md.
The summary can be refreshed after author review without regenerating geometry.

`--cell` is an optional fixed board tile size forwarded to the renderer. It adds whitespace
without changing model scale, projection, thresholds or masks; the default remains 190.
A larger tile is useful for long models. Clipping still fails. The batch itself has no
implicit wall deadline: select compute limits and start early enough for review and delivery.

`--search broad` (default) uses diverse broad split alternatives and independent snapped-plane
updates. `--search strict` uses one strictly occupancy-improving split per region and grouped
plane offsets. `--search multi` runs both from the same proposal, with the configured compute
budget independently applied to each trial, and selects automatically per section: a full
part pass outranks a failure, then lower unchanged formal loss wins; ties keep the first trial.
Tube searches are identical between these neighborhoods and run once. `searches.json` retains
every trial, `search_selection.json` records the choice, and checkpoints retain the best
observed part across trials. The assembled silhouette is evaluated after automatic selection;
part passes alone never establish vehicle acceptance. `fit:*` timings include proposal and all
`search:*` rows and must not be summed with those nested rows.

## Preserve better automatic candidates

```bash
pnpm exec tsx scripts/remaster/cad/select.ts \
  --candidate "$NEW_AUTOMATIC_DELIVERY" --candidate "$PREVIOUS_AUTOMATIC_DELIVERY" \
  --out "$FRESH_SELECTED_DELIVERY" --cell 256
```

Inputs must be complete automatic deliveries with intact spec/export manifest hashes and
matching model source hash, palette hash, section set, solid lattice, affine, camera and
explicit house colour. The output is fresh and cannot overwrite a candidate. Every eligible
part is remeasured using the unchanged formal surface/silhouette implementation and current
130/35/three-per-tube policy. Selection uses a fixed lexicographic rule: full part PASS, fewer
failed independent gates, smaller worst normalized deviation/IoU shortfall, then the existing
formal diagnostic loss. Ties keep the first input. Expanded/per-tube budget violations are
rejected before ranking. The tool never requests model coordinates or a hand-selected part.

`selection.json` and `selection_evaluations.json` retain all origins, current measurements,
rejections and ranks. The assembled silhouette and all boards are regenerated after automatic
selection. This does not turn part passes into a vehicle pass, certify the look, or prove a
global optimum. Upstream trial minutes and selection wall time remain separate in HOURS.md.
The generic `hardSurfaceSelection.ts` owns measurement/ranking; the CLI owns source
compatibility, research-camera replay and game budgets.
