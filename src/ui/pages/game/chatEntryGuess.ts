export interface ChatEntryPolicy {
  readonly open: readonly string[];
  readonly close: readonly string[];
}

/** Deterministic input-only estimate; native text-entry state is not observable here. */
export function createChatEntryGuess(policy: ChatEntryPolicy) {
  let open = false;
  return {
    isOpen: () => open,
    forwardedKeyDown(code: string, repeat = false): void {
      if (repeat) return;
      if (open) {
        if (policy.close.includes(code)) open = false;
      } else if (policy.open.includes(code)) open = true;
    },
    reset(): void {
      open = false;
    },
  };
}

export type ChatEntryGuess = ReturnType<typeof createChatEntryGuess>;
