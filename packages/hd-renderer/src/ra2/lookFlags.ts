/** Every experimental world look needs category metadata to protect resource overlays. */
export function needsSurfaceClasses(params: URLSearchParams): boolean {
  return (
    ['hd-mask', 'hd-sprite', 'hd-blend', 'hd-veh', 'hd-bld', 'hd-bld-polish'].some((key) => params.get(key) === '1') ||
    (params.has('hd-tone') && params.get('hd-tone') !== 'off')
  );
}
