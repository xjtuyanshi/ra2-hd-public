#ifndef NATIVE_HOST_MOVIE_COPY_H
#define NATIVE_HOST_MOVIE_COPY_H

#include <stdint.h>

/* YR 1.001 starts a menu panel clip (the 632x570 Bink behind single-player, skirmish and similar menus) by calling
 * BinkCopyToBuffer one to three times right after BinkOpen and before the first BinkDoFrame. Bink's frame buffers
 * hold no decoded picture yet, and the copy converts their zeroed YUV planes to RGB565 0x04C0, a flat green that
 * covered the whole left panel for 30-90 ms of every 上一页 (Back) into such a menu (live: Build 13, 2026-11-14,
 * 0x04C0 in the copied surface and in the presented frame). The copy is skipped until the handle has decoded a
 * frame (BinkDoFrame or BinkGoto), so the surface keeps the picture the game drew before; nothing is drawn in its
 * place and playback, timing and the first decoded frame are unchanged. Handles the agent does not track (opened
 * before it was configured, or with failed bookkeeping) are copied as before.
 *
 * The skip is bounded: a handle that is still undecoded NH_MOVIE_UNDECODED_COPY_LIMIT_US after its open is copied
 * as the game asked, so a caller that copies without decoding through the hooked imports can at worst lose that
 * short window, never stay blank. The observed early copies all came within 90 ms of the open; the limit leaves room for a loaded machine. */
#define NH_MOVIE_UNDECODED_COPY_LIMIT_US 500000u

enum nh_movie_copy_decision
{
    NH_MOVIE_COPY = 0,          /* Untracked or decoded: the game's own copy. */
    NH_MOVIE_COPY_SKIP = 1,     /* Tracked, undecoded, inside the limit: keep the previous surface picture. */
    NH_MOVIE_COPY_LATE = 2,     /* Tracked, still undecoded past the limit: copy as the game asked. */
};

static inline enum nh_movie_copy_decision nh_movie_copy_decide(int tracked, int decoded, uint64_t since_open_us)
{
    if (!tracked || decoded) return NH_MOVIE_COPY;
    return since_open_us < NH_MOVIE_UNDECODED_COPY_LIMIT_US ? NH_MOVIE_COPY_SKIP : NH_MOVIE_COPY_LATE;
}

/* Bookkeeping for one tracked handle; the agent keeps these in its locked open-handle list. */
struct nh_movie_copy_track
{
    int decoded;
};

static inline void nh_movie_copy_opened(struct nh_movie_copy_track *track)
{
    track->decoded = 0;
}

/* BinkDoFrame and BinkGoto both leave a decoded frame in the handle's buffers. */
static inline void nh_movie_copy_decoded(struct nh_movie_copy_track *track)
{
    track->decoded = 1;
}

#endif
