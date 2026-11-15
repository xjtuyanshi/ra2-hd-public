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

Inputs must be complete automatic deliveries with intact spec/export (and optional search-log) manifest hashes and
matching model source hash, palette hash, section set, solid lattice, affine, camera and
explicit house colour. The output is fresh and cannot overwrite a candidate. Every eligible
part is remeasured using the unchanged formal surface/silhouette implementation and current
130/35/three-per-tube policy. Selection uses a fixed lexicographic rule: full part PASS, fewer
failed independent gates, smaller worst normalized deviation/IoU shortfall, then the existing
formal diagnostic loss. Ties keep the first input. Expanded/per-tube budget violations are
rejected before ranking. The tool never requests model coordinates or a hand-selected part.

Complete `searches.json` states are eligible alongside each delivery's default spec, so the
outer ranking cannot lose a better automatic strict/broad trial. Identical part payloads are
measured once with every alias retained. Original dirty/changed-source provenance is copied
as evidence rather than represented as an immutable revision.

`selection.json` and `selection_evaluations.json` retain all origins, variants, current measurements,
rejections and ranks. The assembled silhouette and all boards are regenerated after automatic
selection. This does not turn part passes into a vehicle pass, certify the look, or prove a
global optimum. Upstream trial minutes and selection wall time remain separate in HOURS.md.
The generic `hardSurfaceSelection.ts` owns measurement/ranking; the CLI owns source
compatibility, research-camera replay and game budgets.

## Source-supported curved reconstruction

The generic `hardSurfaceRound.ts` builds arbitrary-axis circular frusta, polyline cylinder
sweeps with flush bisector joints, bounded torus chords, capsules, spherical/ellipsoidal caps,
and clipped edge fillets. These are ordinary explicit convex halfspace patches with analytic
smooth normals; they use the same `buildPart`, rasterizer and unchanged surface metrics as
plates. Every sweep/torus chord and every carve piece counts toward the expanded budget.
There is no compound primitive that hides a per-voxel cover. New `round` shading metadata is
validated by the existing spec parser and reflected with its geometry. Existing specs without
that metadata retain their geometry and normals.

`hardSurfaceRoundFit.ts` owns source-only cross-section rays, exposed-arc and radial-residual
checks, stable axial tracks, bounded centreline simplification, coherent taper regression,
spherical/ellipsoidal area-profile regression with explicit shrinking-section evidence, capsule-end and circular-quadrant hypotheses.
A torus hypothesis additionally requires occupied source samples along the intervening arc;
its endpoints alone cannot authorize a bend. Local straight/torus/straight elbow fits also test the intervening
source arc, allowing an offset pipe to contain opposite bends instead of requiring its entire path to
lie on one circle. Torus end planes follow the endpoint tangents to meet adjoining straight pipes flush. Tiny cross sections are explicitly marked
ambiguous: a one-cell rail cannot prove its subvoxel cross-section is circular. Attached
continuations require an established track and occupancy support; flat plates alone are not
positive round evidence. The detector can still miss blended/occluded or rapidly changing
features, and individual source-supported hypotheses may fail formal or visual acceptance.

Seed components `type: 'round'` fit source-supported caps, capsules or fillets; `type: 'pipe'`
fits detected sweeps only when they cover at least 90 percent of the selected source cells.
Insufficient evidence is an error, not an exact-cover fallback. Automatic proposals can opt
into pipe extraction with `roundPipes: true` or `run.ts --round-pipes`. The default startup,
library installation and live-cache paths are unchanged. Curved patches are not split or
independently retuned as planar facets by refinement.

Flat circular end faces cannot seed domes. Cap discovery scans lower local peaks alongside taller
payloads, rather than only a fixed band beneath the global maximum. Longitudinal circular hypotheses
require missing source cross-section corners and occupancy support; a long rectangular box is not
round evidence. An established exposed pipe can additionally continue through attached source sections
with occupied circular interiors and an exposed flank. The upper-pipe replacement carves only above
its centreline planes, preserving the parent material below and avoiding floating seams.

A separate offline round pass preserves an earlier automatic delivery's broad plates and
tries source-supported curved replacements, source-profile circular barrel chains and
coplanar boundary adjustments. Opposite inequalities at a shared BSP seam move together;
round tangent planes are excluded. Source/palette bytes and input manifest hashes are
checked before use. Every part and assembly is remeasured in the original fixed camera;
paint, wear and the exported reference geometry are copied without tuning.

```bash
pnpm exec tsx scripts/remaster/cad/round.ts \
  --input "$PREVIOUS_AUTOMATIC_DELIVERY" --out "$FRESH_OUTPUT" \
  --seconds 1800 --cell 256 --font "$CJK_FONT"
pnpm exec tsx scripts/remaster/cad/roundBatch.ts \
  --ids "$MODEL_IDS" --input-root "$PREVIOUS_UNITS_DIRECTORY" \
  --out "$BATCH_OUTPUT" --concurrency 4 --seconds 1800 --cell 256 --font "$CJK_FONT"
python3 scripts/remaster/cad/overview.py \
  --input "$BATCH_OUTPUT" --output "$BATCH_OUTPUT/overview_10.png" --font "$CJK_FONT"
pnpm exec tsx scripts/remaster/cad/roundSummary.ts "$BATCH_OUTPUT"
```

`roundBatch.ts` retains every actual child exit and serializes status writes. Its output root
may already contain the task brief and research notes, but the batch status and unit outputs
must be fresh. `--seconds` is a per-unit soft search limit; the current formal evaluation and
board delivery finish after it. `timing.json` explicitly records time-budget exhaustion.
Prior automatic baseline labour is not new reconstruction time. Shared development,
diagnosis and actual author review must be recorded independently.

This entry uses the revised research gate: assembled IoU at least 0.97 in all eight facings
is hard; bidirectional p99 at most 1.0 is advisory; barrel per-part IoU at least 0.9 is a
separately reported target. Expanded budgets remain 130/35/three per detected circular tube,
and every segment of one barrel lane must share its detected 3D axis. Radius steps come
from source cross-section changes; no invented muzzle rings, transverse boxes or rectangular
barrel clipping are introduced to improve the score. A missed target remains visible even
when the independent revised gate passes. The legacy commands above retain their original
stricter metric semantics.

`roundSummary.ts` requires complete eight-view measurements, actual mesh counts, successful
process evidence, and a `visual_review.json` with `pass`, `reason`, `agentDirectedMinutes`,
`sourceNothingAdded`, `proportionsPreserved`, and `scalesOpened` containing `1x`, `2x`, and
`turntable_8`. Missing review is UNVERIFIED. Any visibly wrong proportion, added component,
blocky round feature or stair-stepped plate fails the visual gate even if assembly IoU passes.
The overview crops native facing-zero tiles without resizing. No offline board establishes
VM, GPU, runtime, performance, hardware or owner artistic acceptance.

### Preserve the hard floor while polishing existing automatic geometry

`round.ts --mode polish` (also forwarded by `roundBatch.ts`) remeasures the earlier delivery,
then attempts compatible convex-neighbour coalescing and source-seeded upper ridges. The
coalescer checks opposing shared halfspaces and every neighbour vertex against all remaining
constraints, with a `1e-7` plane tolerance. Concave unions and incompatible opaque paint,
groove or chamfer metadata cannot merge. Existing subtraction is retained; expanded flat
pieces are exempted from being cut a second time. Round primitives keep their original owner.
This reduces genuine geometric complexity and false internal bevel seams, not voxel count.
All resulting beveled meshes still receive the unchanged formal metrics and silhouette tests.

Ridges fit a circular upper cross-section against quantized source heights, including a
sloping parent surface. Constant-radius runs are split when no radius fits every source
profile; each run needs six source slices. A flat, featureless plate is not a ridge seed.
Only material above the fitted parent plane is replaced. The hidden radius and narrow
cross-section remain ambiguous at low voxel resolution; this is source-supported geometry,
not semantic recognition of pipes or proof of a hidden assembly. Every proposal still has
to preserve the assembly hard floor, budgets, source proportions and actual 1x visual review.

Polish may first search for a small silhouette margin above the hard floor, using up to 40
percent of its compute window. This `0.971` internal search aim is not an acceptance threshold:
the revised hard gate stays at `0.97`, and p99 remains advisory. Coalescing may slightly alter
seeded triangle-sampling statistics even when the un-beveled union is unchanged; its actual
metrics are retained, not replaced with earlier numbers. Barrel specs are remeasured unchanged
in this mode, since the round pass already performed their independent circular-chain search.

For each ridge candidate, polish can coalesce the new cut fragments before checking its actual
expanded count. This is the same verified convex-union operation, not a compound-count exemption.
`coalescing.json` retains the original neighbour names and every local merge. The ridge height-error
field describes raw source profiles before centreline smoothing; it is a proposal diagnostic, not
the delivered surface p99. Final mesh errors and every silhouette are always remeasured.

`round.ts --mode shoulder` is an additional opt-in source pass. It fits stable
exposed elliptic quarter profiles over at least six source slices, verifies the
occupied interior, and rejects straight chamfers and featureless plates. The
unobserved radius remains ambiguous and is marked in the evidence. An explicit
elliptic extrusion supplies the actual halfspaces and analytic normals; axial caps
and caller clipping planes retain flat normals. Related profiles sharing an axial
range are also proposed as a complete bundle, with all resulting pieces charged
to the same budget and the unchanged assembly floor applied to the complete
candidate. This pass does not infer hidden parts or change paint.

New round evaluation logs use one compact JSON object per line. Historical round
artifacts produced before this correction contain a whitespace-separated stream
of pretty-printed JSON objects; retain their original bytes and use a streaming
JSON decoder when examining that historical evidence.
