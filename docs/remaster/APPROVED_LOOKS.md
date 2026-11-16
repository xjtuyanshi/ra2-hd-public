# Approved looks (owner sign-off register)

This register lists the HD looks the owner has approved as final for now ("定稿"). An entry here means the look is the
owner-launcher default and should not be redesigned without a new owner decision; bug fixes and performance work are
still allowed. Work-in-progress looks live on `wip/*` branches and are not listed.

All entries are code and shader parameters in this repository. No game-derived art (HD sprite pages, terrain images,
overlay farms) is stored in Git; those stay in the player's local `gameres/` folders (see [PLAN.md](PLAN.md) and
[AGENTS.md](../../AGENTS.md)).

| Look                                              | Launcher option                                  | Code                                                                           | Approved                               | Notes                                                                                                           |
| ------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------ | -------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Clear modern water (day)                          | `hd-water=clear`                                 | `packages/hd-renderer/src/ra2/clearWater.ts`, `waterLook.ts`, `waterStyles.ts` | 2026-11-13; confirmed final 2026-11-15 | Procedural (no textures); reflections, clear shallows.                                                          |
| Water follows the scene light (night, snow)       | part of `hd-water=clear`                         | `clearWater.ts`, `waterLook.ts` (grade input), RA2 policy in `ra2/`            | 2026-11-15                             | Night maps darken to the original navy with a faint moon sheen; snow sea deeper and colder; day maps unchanged. |
| Drifting cloud shade, deep preset, land and water | `hd-alive=1`, `alive-preset=deep`                | `packages/hd-renderer/src/livingWorld.ts`, `livingWorldNodes.ts`               | 2026-11-14                             | With `hd-tone=lush2` on, lush2 drives the cloud patches at no less than the deep preset's depth.                |
| "润" lush ground grade, Sunday draft              | `hd-tone=lush2` (shore band kept, `shoreKeep` 1) | `packages/hd-renderer/src/ra2/lush2Tone.ts`, `lush2Tuning.ts`                  | 2026-11-15                             | Saturation lowered from r3; ore, gems and house colours unchanged; tuning panel only with `hd-tune=1`.          |

The authoritative list of launcher defaults is `scripts/remaster/play-options.mjs` (`HD_PLAY_OPTIONS`).
