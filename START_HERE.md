# Start here: the current HD remaster on a friend's Windows PC

Give your Windows coding assistant this file and one sentence:

> 请阅读 START_HERE.md，从公开镜像下载并复用现成的第 13 版高清素材，在我的 Windows 电脑上完成适配和实机验证；只修改我自己的副本，不向原作者仓库推送。

## Scope and current version

The current PC owner is the original author's friend. Work only in that PC owner's local
copy and a separate Windows branch. Do not push to the original repository, open a PR,
change visibility or upload resources. Only use the PC owner's own GitHub identity.
Original-author instructions in older runbooks to sign in as `xjtuyanshi` or push every
step to the original `main` do not apply. Follow the remaining [AGENTS.md](AGENTS.md) rules.

The private development repository is **`xjtuyanshi/ra2-hd-assets`**. All downloads for
this handoff come from **`xjtuyanshi/ra2-hd-public`**, the public sharing mirror, even
when your own `origin` points to a fork. Never request private-repository credentials.
Mirroring is on demand, not automatic; see [PUBLIC_MIRROR.md](PUBLIC_MIRROR.md).

Use public `main`, including the **Build 13 Sunday** defaults approved on 2026-11-15:
clear water that follows scene lighting, deep drifting cloud shade and `lush2` ground
grading. See [approved looks](docs/remaster/APPROVED_LOOKS.md). Experimental `wip/*`
branches are preserved for reference; their unpublished local art is not implied to be
in the released pack. **Windows still needs a port and real-PC acceptance**; follow
[WINDOWS_SETUP.md](WINDOWS_SETUP.md), not the historical Mac launch commands.

## All resource locations

Clone retrieves code, not release attachments. The "HD cache" or "farm" is the
**already-produced HD resource pack**: download, extract and verify it, **do not remake art**.
All release tags below belong to `xjtuyanshi/ra2-hd-public`.

| Item                            | Location                                                                                                                         | Action                                                                                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source and renderer             | Public `main`                                                                                                                    | Clone to `C:\ra2-hd\code`; record the commit.                                                                                                                                   |
| Current HD increment            | `farm-build13-sunday-20261115`                                                                                                   | Download `hd-cache-play12-b13f.manifest.json`, this release's `restore_farm.py`, `SHA256SUMS` and `hd-cache-play12-b13f.part00.tar.zst`.                                        |
| Required base HD content        | `farm-build12-20261102`                                                                                                          | Also obtain all 32 `hd-cache-play12.part00.tar.zst` through `part31.tar.zst` chunks, one at a time if space is limited. The increment alone is not a complete pack.             |
| Historical bridge binaries      | `farm-build12-20261102`: `ddraw.dll`, `ra2launch.exe`                                                                            | These are project-built 32-bit Windows components, not original game files. They are Build 12 outputs: check compatibility and rebuild current source when needed for Build 13. |
| Current bridge source           | `packages/native-host/`, `build.sh`, `launcher/build.sh`                                                                         | Inspect compiler and Go requirements and adapt build invocation to Windows when rebuilding. Do not claim old binaries were rebuilt for current code.                            |
| Original game                   | The PC owner's legally held installation                                                                                         | Copy to `C:\ra2-hd\game\YURI`. Never download original game executables or resources.                                                                                           |
| Optional art-making environment | `env-build12-20261102`: six `env-build12.part00.tar.zst` through `part05.tar.zst` chunks, manifest, restore script and checksums | Not needed to play or port the launcher. Skip Wine, SDXL and ControlNet. Notices are in [backup/LICENSES](backup/LICENSES/README.md).                                           |

The current pack has 6,387 unique blobs and 13,216 paths: 65,675,336,918 unique bytes
(about 61.2 GiB). The increment supplies 148 new blobs (about 1.9 GB); the base supplies
the other 6,239. Hard links deduplicate paths; copies can increase the pack to about
98.2 GB, before game/code/staging space.

## Restore the current pack

Use **`restore_farm.py` from the Build 13 release**: it supports incremental `/2`
manifests. The older Build 12 release/repository script does not. The archived manifest
names the private repository, so **always pass `--repo xjtuyanshi/ra2-hd-public`**,
including with `--fetch`, to resolve new and base chunks publicly. Preserve manifest
and chunk bytes/checksums.

Verify the Build 13 script and manifest against its `SHA256SUMS`. Translate this reference
sequence to the PC's PowerShell/Python (`python` may be `py -3`):

```text
python restore_farm.py hd-cache-play12-b13f.manifest.json NEW TARGET --chunks 0 --base-chunks none --repo xjtuyanshi/ra2-hd-public
python restore_farm.py hd-cache-play12-b13f.manifest.json NEW TARGET --chunks none --base-chunks farm-build12-20261102:N --base-chunk-dir BASE --repo xjtuyanshi/ra2-hd-public
python restore_farm.py hd-cache-play12-b13f.manifest.json - TARGET --finish --repo xjtuyanshi/ra2-hd-public
```

`NEW` holds the incremental release files, `BASE` holds the current base chunk, and
`TARGET` is `C:\ra2-hd\farm\hd-cache-play12-b13f`. Run the middle command once per
base chunk, with decimal `N` from 0 to 31; filenames still use 00–31. The script can
download missing chunks with `--fetch` through GitHub CLI, or download via anonymous
HTTPS first. Base hashes are embedded in the `/2` manifest. Delete a downloaded chunk
only after successful restore and verification. A missing hash or failed check is not
success. Final verification must report `13216 farm paths checked, 0 problems`.

Test Windows path handling with the increment first; explicitly validate hard links and
final symbolic links. Windows restoration remains unverified. Point `RA2_HD_CACHE_DIR`
at the restored Build 13 target. The base release's `launcher-build12.command` is a Mac
reference, not a Windows entry.

## Implement and verify

1. Check public source/release access and free disk before large downloads. Public HTTPS
   needs no account; if GitHub CLI requests login, use the PC owner's own account or
   anonymous HTTPS instead.
2. Read the Windows guide and inspect actual tool requirements. Initialize local Git if
   using a ZIP without `.git`, then run `git config core.hooksPath scripts/git-hooks`,
   `pnpm install --frozen-lockfile` and `pnpm run check`. Print software installation,
   login, administrator and DLL-registration commands for the PC owner to execute;
   preserve system security settings.
3. Restore the pack, implement Windows native startup and a double-click entry, and
   preserve Mac behavior and exact backup/restore of the game's `ddraw.dll`.
4. Run `pnpm run check` and relevant regressions from [docs/TESTING.md](docs/TESTING.md).
   Exercise menus, HD/original switching, selection, movement, building, scrolling,
   save/load and exit restoration. The PC owner plays a full match and checks the other
   Windows-guide acceptance items. Report passed, failed and `UNVERIFIED` separately,
   in Chinese. Keep game files, HD resources, models, recordings and personal data out
   of Git; preserve EA and third-party notices.

[AI_SETUP.md](AI_SETUP.md) and [RESTORE.md](RESTORE.md) are historical Build 12 Mac
references for background/base inventory, not replacements for the incremental restore above.
