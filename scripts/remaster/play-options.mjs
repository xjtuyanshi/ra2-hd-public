import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

// Both owner launchers use this exact look. Resolution belongs to each backend:
// v86 can measure the page at boot; a native process reads its INI before start.
export const HD_PLAY_OPTIONS = Object.freeze({
  hd: '1',
  'hd-look': 'cinematic',
  'hd-fx': 'enhanced',
  'hd-water': 'clear',
  'hd-vlook': 'remaster',
  'hd-vproj': 'vehicles-hd',
  'hd-ui': '1',
  // Owner choice 2026-11-14: drifting cloud shade on land and water, deeper preset.
  'hd-alive': '1',
  'alive-preset': 'deep',
  // Owner choice 2026-11-15 (Sunday build): the lush2 ground grade (“润” third draft, shoreKeep 1 keeps the shore band
  // as Build 13 draws it). lush2 drives its grade with its own cloud patches, which replace the deep preset above
  // while it is on; hd-tone=off restores the deep clouds. The tuning panel stays behind hd-tune=1.
  'hd-tone': 'lush2',
  // Single-pass reconstruction of the original menus/dialogs; HD passthrough
  // keeps its separate 1:1 overlay path and does not run this filter.
  sr: '1',
});

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  console.log(new URLSearchParams(HD_PLAY_OPTIONS).toString());
