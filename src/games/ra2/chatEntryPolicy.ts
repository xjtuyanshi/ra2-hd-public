/** Shared RA2/YR default keyboard policy; remapped KEYBOARD.INI bindings can desynchronise the estimate. */
export const RA2_CHAT_ENTRY_POLICY = {
  open: ['Enter', 'NumpadEnter', 'Backspace'],
  close: ['Enter', 'NumpadEnter', 'Escape'],
} as const;
