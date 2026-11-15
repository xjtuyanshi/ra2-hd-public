import type { HdRenderer } from '../HdRenderer';
import type { AliveOptions } from '../livingWorld';
import { mountTuningPanel } from '../ui/tuningPanel';
import { applyLush2, lush2Defaults, LUSH2_SCHEMA } from './lush2Tone';

/** A bridge/bench owns this controller; current means exactly no tone, with the entry's original alive options. */
export function mountLush2Tuning(
  host: HTMLElement,
  hd: HdRenderer,
  theater: string,
  currentAlive: AliveOptions,
  returnFocus?: () => void,
  textEntryActive?: () => boolean,
) {
  let values = lush2Defaults(theater);
  let enabled = true;
  const apply = () => {
    if (enabled) hd.setAlive(applyLush2(hd.settings, theater, values));
    else {
      hd.settings.groundTone = null;
      hd.settings.tone = null;
      hd.settings.cloudTone = null;
      hd.setAlive(currentAlive);
    }
    hd.applySettings();
  };
  const panel = mountTuningPanel(host, {
    returnFocus,
    textEntryActive,
    schema: LUSH2_SCHEMA,
    recommended: lush2Defaults(theater),
    storageKey: `hd-lush2-v5:${theater}`,
    get: () => ({ ...values }),
    set: (next) => {
      values = { ...next };
    },
    ab: (next) => {
      enabled = next;
      apply();
    },
  });
  return panel;
}
