import type { ExportOptions } from '@neuroloom/core';
import { tr } from '../i18n';
import { useStudio } from '../store';

type Flag = Exclude<keyof ExportOptions, 'nesting'>;

/** Options that are on unless turned off. */
const ON_BY_DEFAULT: Flag[] = ['shapeComments', 'typeHints'];

const FLAGS: { key: Flag; zh: string; en: string; detailZh: string; detailEn: string }[] = [
  { key: 'shapeComments', zh: '形状注释', en: 'Shape comments', detailZh: 'forward() 每行后面写上 # [B, 64, 56, 56]。', detailEn: 'Each forward() line ends with # [B, 64, 56, 56].' },
  { key: 'typeHints', zh: '类型标注', en: 'Type hints', detailZh: 'x: torch.Tensor、-> torch.Tensor 和超参数的类型。', detailEn: 'x: torch.Tensor, -> torch.Tensor and hyperparameter types.' },
  {
    key: 'sequentialRepeats',
    zh: '重复块写成 nn.Sequential',
    en: 'Repeats as nn.Sequential',
    detailZh: '单输入单输出的重复块写成 nn.Sequential 并直接调用，不用 nn.ModuleList 加 for 循环。权重键名相同。',
    detailEn: 'Repeats with one input and one output become an nn.Sequential called once instead of an nn.ModuleList and a for loop. Weight names stay the same.',
  },
  { key: 'docstrings', zh: '类说明', en: 'Class docstrings', detailZh: '每个类带一行 docstring，写明参数量。', detailEn: 'Each class gets a one-line docstring with its parameter count.' },
  { key: 'header', zh: '文件头注释', en: 'File header', detailZh: '文件开头注明项目名和来源。', detailEn: 'A comment at the top naming the project.' },
  {
    key: 'selfTest',
    zh: '自测片段',
    en: 'Self-test',
    detailZh: 'if __name__ == "__main__": 用输入节点的形状（B = 1，其他符号 128）跑一次前向，并断言输出形状。',
    detailEn: 'if __name__ == "__main__": runs one forward pass on the input shapes (B = 1, other symbols 128) and asserts the output shapes.',
  },
  {
    key: 'splitFiles',
    zh: '每个类一个文件',
    en: 'One file per class',
    detailZh: '导出为一个 Python 包：每个类一个模块，加上 __init__.py。桌面版选择文件夹，网页版下载 .zip。',
    detailEn: 'Export a Python package: a module per class plus __init__.py. The desktop app asks for a folder; the web version downloads a .zip.',
  },
];

const NESTING = [
  { value: 0, zh: '每行一步', en: 'One step per line' },
  { value: 1, zh: '最多嵌套 1 层', en: 'Nest up to 1 call' },
  { value: 2, zh: '最多嵌套 2 层', en: 'Nest up to 2 calls' },
  { value: 3, zh: '最多嵌套 3 层', en: 'Nest up to 3 calls' },
];

/** Selected when the project has no options; one object, so the selection stays the same between renders. */
const NONE: ExportOptions = {};

/** Normalizes options: defaults are left out, so the project file only records changes. */
function normalized(options: ExportOptions): ExportOptions {
  const out: ExportOptions = {};
  for (const f of FLAGS) {
    const value = options[f.key];
    if (value !== undefined && value !== ON_BY_DEFAULT.includes(f.key)) out[f.key] = value;
  }
  if (options.nesting) out.nesting = options.nesting;
  return out;
}

/** Export options (PRD F5.3), kept in the project; the code preview follows them as they change. */
export function ExportOptionsDialog({ onClose }: { onClose: () => void }) {
  const options = useStudio((s) => s.project.exportOptions ?? NONE);
  const setExportOptions = useStudio((s) => s.setExportOptions);
  const update = (patch: ExportOptions) => setExportOptions(normalized({ ...options, ...patch }));
  const isOn = (key: Flag) => options[key] ?? ON_BY_DEFAULT.includes(key);

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-options-title"
        className="flex max-h-full w-[min(560px,100%)] flex-col rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="border-b border-line px-5 py-4">
          <h2 id="export-options-title" className="text-[15px] font-semibold">
            {tr('导出选项', 'Export options')}
          </h2>
          <p className="mt-1 text-[12px] text-muted">
            {tr('只改变代码的写法，不改变模型。选项保存在项目里，下方的代码预览会立即更新。', 'These change how the code is written, never the model. They are saved with the project, and the code preview updates as you go.')}
          </p>
        </header>
        <div className="min-h-0 flex-1 space-y-3 overflow-auto px-5 py-4">
          {FLAGS.map((f) => (
            <label key={f.key} className="flex cursor-pointer gap-3">
              <input type="checkbox" className="mt-0.5" checked={isOn(f.key)} onChange={(e) => update({ [f.key]: e.target.checked })} />
              <span>
                <span className="block text-[13px] font-medium">{tr(f.zh, f.en)}</span>
                <span className="block text-[12px] leading-relaxed text-muted">{tr(f.detailZh, f.detailEn)}</span>
              </span>
            </label>
          ))}
          <label className="flex items-start gap-3">
            <span className="w-4" />
            <span className="flex-1">
              <span className="block text-[13px] font-medium">{tr('嵌套调用', 'Nested calls')}</span>
              <span className="mb-1.5 block text-[12px] leading-relaxed text-muted">
                {tr('只用一次的结果可以直接写进下一步的调用，如 x = F.relu(self.bn1(self.conv1(x)))。', 'A result used once can go straight into the next call, as in x = F.relu(self.bn1(self.conv1(x))).')}
              </span>
              <select
                value={options.nesting ?? 0}
                onChange={(e) => update({ nesting: Number(e.target.value) })}
                className="rounded-md border border-line bg-bg px-2 py-1 text-[12.5px] outline-none focus:border-primary"
              >
                {NESTING.map((n) => (
                  <option key={n.value} value={n.value}>
                    {tr(n.zh, n.en)}
                  </option>
                ))}
              </select>
            </span>
          </label>
        </div>
        <footer className="flex justify-between gap-2 border-t border-line px-5 py-3">
          <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2 disabled:opacity-40" disabled={Object.keys(options).length === 0} onClick={() => setExportOptions({})}>
            {tr('恢复默认', 'Reset to defaults')}
          </button>
          <button type="button" className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover" onClick={onClose}>
            {tr('完成', 'Done')}
          </button>
        </footer>
      </div>
    </div>
  );
}
