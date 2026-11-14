/** Development CPU diagnostics. GPU execution/upload time requires a separate timestamp measurement. */
export type LightCostPart = 'contactRasterUpload' | 'fieldUpdate' | 'emissiveBuild' | 'physicalLampUpdate';
export class LightCost {
  private readonly pending: Record<LightCostPart, number> = {
    contactRasterUpload: 0,
    fieldUpdate: 0,
    emissiveBuild: 0,
    physicalLampUpdate: 0,
  };
  private readonly frames: Record<LightCostPart, number>[] = [];
  private readonly renders: number[] = [];
  lightsSplatted = 0;
  texelsSplattedEstimate = 0;
  record(part: LightCostPart, ms: number): void {
    if (import.meta.env.DEV) this.pending[part] += Math.max(0, ms);
  }
  measure<T>(part: LightCostPart, work: () => T): T {
    if (!import.meta.env.DEV) return work();
    const start = performance.now();
    try {
      return work();
    } finally {
      this.record(part, performance.now() - start);
    }
  }
  finishFrame(rendered: boolean, now = performance.now()): void {
    if (!import.meta.env.DEV) return;
    this.frames.push({ ...this.pending });
    if (this.frames.length > 120) this.frames.shift();
    for (const key of Object.keys(this.pending) as LightCostPart[]) this.pending[key] = 0;
    if (rendered) this.renders.push(now);
    while (this.renders.length && this.renders[0]! <= now - 1000) this.renders.shift();
  }
  snapshot() {
    const msPerFrame = { ...this.pending };
    for (const key of Object.keys(msPerFrame) as LightCostPart[])
      msPerFrame[key] = this.frames.reduce((sum, f) => sum + f[key], 0) / Math.max(1, this.frames.length);
    return {
      msPerFrame,
      samples: this.frames.length,
      lightsSplatted: this.lightsSplatted,
      texelsSplattedEstimate: this.texelsSplattedEstimate,
      fieldRendersPerSecond: this.renders.filter((t) => t > performance.now() - 1000).length,
    };
  }
}
