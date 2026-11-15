#ifndef NATIVE_HOST_MOVIE_COPY_H
#define NATIVE_HOST_MOVIE_COPY_H

/* YR 1.001 starts a menu panel clip (the 632x570 Bink behind single-player, skirmish and similar menus) by calling
 * BinkCopyToBuffer one to three times right after BinkOpen and before the first BinkDoFrame. Bink's frame buffers
 * hold no decoded picture yet, and the copy converts their zeroed YUV planes to RGB565 0x04C0, a flat green that
 * covered the whole left panel for 30-90 ms of every 上一页 (Back) into such a menu (live: Build 13, 2026-11-14,
 * 0x04C0 in the copied surface and in the presented frame). The copy is skipped until the handle has decoded a
 * frame (BinkDoFrame or BinkGoto), so the surface keeps the picture the game drew before; nothing is drawn in its
 * place and playback, timing and the first decoded frame are unchanged. Handles the agent does not track (opened
 * before it was configured, or with failed bookkeeping) are copied as before. */
static inline int nh_movie_copy_skipped(int tracked, int decoded)
{
    return tracked && !decoded;
}

#endif
