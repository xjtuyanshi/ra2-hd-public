import { describe, expect, it } from 'vitest';
import { createChatEntryGuess } from '../../src/ui/pages/game/chatEntryGuess';
import { RA2_CHAT_ENTRY_POLICY } from '../../src/games/ra2/chatEntryPolicy';

describe('forwarded-key chat estimate', () => {
  it.each(RA2_CHAT_ENTRY_POLICY.open)('opens with %s and closes with either Enter or Escape', (open) => {
    const chat = createChatEntryGuess(RA2_CHAT_ENTRY_POLICY);
    for (const close of RA2_CHAT_ENTRY_POLICY.close) {
      chat.forwardedKeyDown(open);
      expect(chat.isOpen()).toBe(true);
      for (const code of ['Backspace', 'Backquote', 'KeyA']) chat.forwardedKeyDown(code);
      expect(chat.isOpen()).toBe(true);
      chat.forwardedKeyDown(close);
      expect(chat.isOpen()).toBe(false);
    }
    chat.forwardedKeyDown('Escape');
    expect(chat.isOpen()).toBe(false);
  });

  it('ignores repeats in either state and resets explicitly', () => {
    const chat = createChatEntryGuess(RA2_CHAT_ENTRY_POLICY);
    chat.forwardedKeyDown('Enter', true);
    expect(chat.isOpen()).toBe(false);
    chat.forwardedKeyDown('Enter');
    for (const code of RA2_CHAT_ENTRY_POLICY.close) chat.forwardedKeyDown(code, true);
    expect(chat.isOpen()).toBe(true);
    chat.reset();
    expect(chat.isOpen()).toBe(false);
  });

  it('injects bindings without assuming game-specific keys', () => {
    const chat = createChatEntryGuess({ open: ['KeyT'], close: ['KeyQ'] });
    chat.forwardedKeyDown('Enter');
    expect(chat.isOpen()).toBe(false);
    chat.forwardedKeyDown('KeyT');
    expect(chat.isOpen()).toBe(true);
    chat.forwardedKeyDown('Escape');
    expect(chat.isOpen()).toBe(true);
    chat.forwardedKeyDown('KeyQ');
    expect(chat.isOpen()).toBe(false);
  });
});
