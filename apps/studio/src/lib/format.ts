export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/** FLOPs with a decimal prefix: 1.82 GFLOPs. */
export function formatFlops(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOPs`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GFLOPs`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MFLOPs`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KFLOPs`;
  return `${n} FLOPs`;
}

/** Bytes in binary units, as memory is usually reported: 44.6 MB. */
export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}
