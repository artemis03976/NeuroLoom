// Upsampling (PRD appendix: 上采样与填充).
import { msg } from '../i18n';
import { dimMul, UNKNOWN, type Dim } from '../shape';
import { fail, IN, L, lit, num, ok, OUT, P, rankError, resolveParams } from './helpers';
import type { NodeSpec } from './types';

const upsample: NodeSpec = {
  type: 'torch.nn.Upsample',
  kind: 'layer',
  category: 'conv',
  displayName: L('Upsample', '上采样'),
  summary: L('Resizes spatial dimensions by a scale factor or to a fixed size.', '按倍数或到固定尺寸缩放空间维度。'),
  docUrl: 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html',
  params: [
    P.ints('size', '目标尺寸', { nullable: true, default: null }),
    P.float('scale_factor', '缩放倍数', { nullable: true, default: null, initial: 2.0 }),
    P.enum('mode', '插值方式', { default: 'nearest', options: ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] }),
  ],
  inputs: IN,
  outputs: OUT,
  defaultName: 'upsample',
  infer([x], params) {
    const bad = rankError(x!, [3, 4, 5], 'Upsample');
    if (bad) return fail(bad);
    const n = x!.shape.length - 2;
    const size = lit(params, 'size');
    const scale = lit(params, 'scale_factor');
    if ((size === null) === (scale === null)) return fail(msg('size 和 scale_factor 必须且只能设置一个', 'set exactly one of size and scale_factor'));
    let spatial: Dim[];
    if (size !== null) {
      spatial = Array.isArray(size) ? (size as number[]) : Array<number>(n).fill(size as number);
      if (spatial.length !== n) return fail(msg(`size 需要 ${n} 个值`, `size needs ${n} values`));
    } else {
      const s = scale as number;
      spatial = x!.shape.slice(2).map((d) =>
        typeof d === 'number' ? Math.floor(d * s) : Number.isInteger(s) ? dimMul(d, s) : UNKNOWN,
      );
    }
    return ok([{ shape: [x!.shape[0]!, x!.shape[1]!, ...spatial], dtype: x!.dtype }], resolveParams(params));
  },
  shapeDoc: L(
    'H_out = size, or ⌊H · scale_factor⌋ (likewise for every spatial dimension); N and C are unchanged',
    'H_out = size，或 ⌊H · scale_factor⌋（每个空间维同理）；N 与 C 不变',
  ),
  describe: (r) => (r.scale_factor !== null ? `×${r.scale_factor} · ${r.mode}` : `→ ${r.size} · ${r.mode}`),
  codegen: { ctor: 'nn.Upsample' },
  example: { inputs: [{ shape: 'B, 3, 4, 4' }], params: { scale_factor: 2.0, mode: 'bilinear' } },
};

const pixelShuffle: NodeSpec = {
  type: 'torch.nn.PixelShuffle',
  kind: 'layer',
  category: 'conv',
  displayName: L('PixelShuffle', '像素重排'),
  summary: L('Moves channels into space: [C·r², H, W] → [C, H·r, W·r].', '把通道搬到空间维：[C·r², H, W] → [C, H·r, W·r]。'),
  docUrl: 'https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html',
  params: [P.int('upscale_factor', '放大倍数 r', { initial: 2, min: 1 })],
  inputs: IN,
  outputs: OUT,
  defaultName: 'shuffle',
  infer([x], params) {
    const bad = rankError(x!, [4], 'PixelShuffle');
    if (bad) return fail(bad);
    const r = num(params, 'upscale_factor');
    const [b, c, h, w] = x!.shape as [Dim, Dim, Dim, Dim];
    if (typeof c === 'number' && c % (r * r) !== 0) {
      return fail(msg(`通道数 ${c} 不能被 r²=${r * r} 整除`, `the number of channels (${c}) is not divisible by r²=${r * r}`));
    }
    return ok(
      [{ shape: [b, typeof c === 'number' ? c / (r * r) : UNKNOWN, dimMul(h, r), dimMul(w, r)], dtype: x!.dtype }],
      resolveParams(params),
    );
  },
  shapeDoc: L('[N, C·r², H, W] → [N, C, H·r, W·r], r = upscale_factor', '[N, C·r², H, W] → [N, C, H·r, W·r]，r = upscale_factor'),
  describe: (r) => `×${r.upscale_factor}`,
  codegen: { ctor: 'nn.PixelShuffle' },
  example: { inputs: [{ shape: 'B, 8, 3, 3' }], params: { upscale_factor: 2 } },
};

export const RESIZE_SPECS: NodeSpec[] = [upsample, pixelShuffle];
