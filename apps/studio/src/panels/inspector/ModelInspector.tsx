// Shown when nothing is selected: the module on the canvas, its hyperparameters
// (PRD 9.3) and its problems, plus project-level Python for the entry model.
import { literalSource, type HyperParam } from '@neuroloom/core';
import { useState } from 'react';
import { fillShapesByRunning, hasHiddenShapes } from '../../commands';
import { tr } from '../../i18n';
import { formatCount } from '../../lib/format';
import { useCurrentModule, useStudio } from '../../store';
import { CodeArea, field, fieldBase, Section, ValueField } from './fields';

function RunButton() {
  const [busy, setBusy] = useState(false);
  return (
    <button
      type="button"
      disabled={busy}
      onClick={() => {
        setBusy(true);
        void fillShapesByRunning().finally(() => setBusy(false));
      }}
      className="rounded-md border border-line px-2.5 py-1 text-[12px] hover:bg-surface-2 disabled:opacity-50"
    >
      {busy ? tr('正在运行…', 'Running…') : tr('运行一次，补全形状', 'Run once to fill in shapes')}
    </button>
  );
}

/** One hyperparameter row; the name is committed on blur, so a rename happens once. */
function HyperParamRow({ hp, onChange, onRemove }: { hp: HyperParam; onChange: (next: HyperParam) => void; onRemove: () => void }) {
  const [name, setName] = useState<string | null>(null);
  const shown = name ?? hp.name;
  const valid = /^[A-Za-z_]\w*$/.test(shown);
  const commitName = () => {
    if (name !== null && valid && name !== hp.name) onChange({ ...hp, name });
    setName(null);
  };
  return (
    <div className="mb-2 rounded-md border border-line p-2">
      <div className="mb-1.5 flex items-center gap-1.5">
        <input
          className={`${fieldBase} min-w-0 flex-1 ${!valid ? 'border-error' : ''}`}
          value={shown}
          aria-label={tr('名称', 'Name')}
          spellCheck={false}
          onChange={(e) => setName(e.target.value)}
          onBlur={commitName}
          onKeyDown={(e) => e.key === 'Enter' && commitName()}
        />
        <input
          className={`${fieldBase} w-20 flex-none`}
          value={hp.annotation ?? ''}
          placeholder={tr('类型', 'type')}
          aria-label={tr('类型注解', 'Annotation')}
          spellCheck={false}
          onChange={(e) => onChange({ ...hp, ...(e.target.value.trim() ? { annotation: e.target.value.trim() } : { annotation: undefined }) })}
        />
        <button type="button" className="flex-none rounded px-1.5 text-[14px] text-muted hover:text-error" title={tr('删除', 'Remove')} onClick={onRemove}>
          ×
        </button>
      </div>
      <ValueField
        value={hp.default}
        placeholder={tr('默认值（留空 = 必填）', 'default (empty = required)')}
        onCommit={(v) => {
          const next: HyperParam = { name: hp.name, ...(hp.annotation ? { annotation: hp.annotation } : {}) };
          if (v && !('auto' in v)) next.default = v;
          onChange(next);
        }}
      />
    </div>
  );
}

export function ModelInspector() {
  const mod = useCurrentModule();
  const project = useStudio((s) => s.project);
  const view = useStudio((s) => s.view);
  const analysis = useStudio((s) => s.viewAnalysis);
  const { setClassName, selectNode, setHyperparams, setEntry, setText } = useStudio.getState();
  const isEntry = project.entry === mod.id;
  const errors = analysis.diagnostics.filter((d) => d.severity === 'error').length;
  const editableHyperparams = !mod.sequential && !mod.opaque;

  const updateHyperparam = (i: number, next: HyperParam) => {
    const old = mod.hyperparams[i]!;
    const list = mod.hyperparams.map((h, j) => (j === i ? next : h));
    setHyperparams(list, old.name !== next.name ? { [old.name]: next.name } : {});
  };
  const addHyperparam = () => {
    let n = mod.hyperparams.length + 1;
    while (mod.hyperparams.some((h) => h.name === `param${n}`)) n++;
    setHyperparams([...mod.hyperparams, { name: `param${n}`, annotation: 'int', default: { literal: 1 } }]);
  };

  // Where the shapes on the canvas come from.
  let context = '';
  if (!isEntry) {
    if (view.path.length > 0) {
      let moduleId = project.entry;
      const names = view.path.map((id) => {
        const node = project.modules[moduleId]?.nodes[id];
        moduleId = node?.type ?? moduleId;
        return node?.name ?? '?';
      });
      context = tr(`显示的形状来自实例 ${names.join('.')}`, `Shapes shown are those of the instance ${names.join('.')}`);
    } else {
      context = tr('这个模块没有被入口模型使用，形状按默认超参数和声明的输入形状推断。', 'The entry model does not use this module; shapes follow its defaults and declared input shapes.');
    }
  }

  return (
    <>
      <Section
        title={isEntry ? tr('入口模型', 'Entry model') : mod.sequential ? 'nn.Sequential' : tr('模块', 'Module')}
        aside={
          !isEntry && !mod.sequential && !mod.opaque ? (
            <button type="button" className="font-normal text-primary hover:underline" onClick={() => setEntry(mod.id)}>
              {tr('设为入口', 'Make entry')}
            </button>
          ) : undefined
        }
      >
        <label className="mb-1 block text-[12px] text-muted">{tr('类名', 'Class name')}</label>
        <input className={field} value={mod.className} disabled={mod.sequential} onChange={(e) => setClassName(e.target.value)} spellCheck={false} />
        {context && <p className="mt-2 text-[11.5px] leading-relaxed text-muted">{context}</p>}
        <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12px]">
          <dt className="text-muted">{tr('节点', 'Nodes')}</dt>
          <dd className="font-mono">{Object.keys(mod.nodes).length}</dd>
          <dt className="text-muted">{tr('参数量', 'Parameters')}</dt>
          <dd className="font-mono">
            {analysis.totalParams.toLocaleString()} <span className="text-muted">({formatCount(analysis.totalParams)})</span>
            {analysis.partial && <span className="ml-1 text-[11px] text-warning">{tr('不含源码/外部模块', 'excluding source/external modules')}</span>}
          </dd>
        </dl>
      </Section>

      {editableHyperparams && (
        <Section
          title={tr(`超参数（${mod.hyperparams.length}）`, `Hyperparameters (${mod.hyperparams.length})`)}
          aside={
            <button type="button" className="font-normal text-primary hover:underline" onClick={addHyperparam}>
              {tr('+ 添加', '+ Add')}
            </button>
          }
        >
          {mod.hyperparams.length === 0 && (
            <p className="text-[11.5px] leading-relaxed text-muted">
              {tr('超参数是 __init__ 的参数。节点参数可以用 ƒx 写成引用它们的表达式，每个实例传入不同的值。', 'Hyperparameters are __init__ arguments. Node parameters can use ƒx to refer to them, and each instance passes its own values.')}
            </p>
          )}
          {mod.hyperparams.map((hp, i) => (
            <HyperParamRow
              key={`${i}:${hp.name}`}
              hp={hp}
              onChange={(next) => updateHyperparam(i, next)}
              onRemove={() => setHyperparams(mod.hyperparams.filter((_, j) => j !== i))}
            />
          ))}
        </Section>
      )}

      {(mod.constants?.length ?? 0) > 0 && (
        <Section title={tr('常量', 'Constants')}>
          <ul className="space-y-0.5 font-mono text-[12px]">
            {mod.constants!.map((c) => (
              <li key={c.name}>
                self.{c.name} = <span className="text-muted">{c.expr}</span>
                {analysis.env[c.name] !== undefined && <span className="ml-1 text-[11px] text-muted">({literalSource(analysis.env[c.name]!)})</span>}
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Section title={tr(`问题（${analysis.diagnostics.length}）`, `Problems (${analysis.diagnostics.length})`)}>
        {analysis.diagnostics.length === 0 ? (
          <p className="text-[12px] text-success">{tr('形状检查通过，可以导出。', 'Shape check passed; ready to export.')}</p>
        ) : (
          <ul className="space-y-1.5">
            {analysis.diagnostics.map((d, i) => (
              <li key={i}>
                <button
                  type="button"
                  disabled={!d.nodeId}
                  onClick={() => d.nodeId && selectNode(d.nodeId)}
                  className={`text-left text-[12px] leading-relaxed ${d.severity === 'error' ? 'text-error' : 'text-warning'} enabled:hover:underline`}
                >
                  {d.nodeId && <span className="font-mono">{mod.nodes[d.nodeId]?.name}：</span>}
                  {d.message}
                </button>
              </li>
            ))}
          </ul>
        )}
        {errors > 0 && (
          <p className="mt-2 text-[11px] text-muted">{tr('有错误时仍可导出，代码中会留下 TODO(neuroloom) 标记。', 'You can still export; the code gets TODO(neuroloom) markers.')}</p>
        )}
      </Section>

      {hasHiddenShapes() && (
        <Section title={tr('代码块与外部模块的形状', 'Shapes of code blocks and external modules')}>
          <p className="mb-2 text-[11.5px] leading-relaxed text-muted">
            {tr(
              '形状推断看不到其中的计算。可以在 Python 子进程中按入口的输入形状运行一次模型，把观察到的形状声明到这些节点上。',
              'Inference cannot see inside them. Run the model once in a Python subprocess with the entry’s input shapes to declare the observed shapes on those nodes.',
            )}
          </p>
          <RunButton />
        </Section>
      )}

      {!mod.sequential && !mod.opaque && (mod.initCode !== undefined || mod.members !== undefined) && (
        <Section title={tr('保留的 Python 代码', 'Kept Python code')}>
          {mod.initCode !== undefined && (
            <>
              <label className="mb-1 block text-[12px] text-muted">{tr('__init__ 末尾', 'End of __init__')}</label>
              <CodeArea value={mod.initCode} rows={4} onCommit={(v) => setText('initCode', v)} />
            </>
          )}
          {mod.members !== undefined && (
            <>
              <label className="mt-2 mb-1 block text-[12px] text-muted">{tr('其他方法与类属性', 'Other methods and class attributes')}</label>
              <CodeArea value={mod.members} rows={6} onCommit={(v) => setText('members', v)} />
            </>
          )}
        </Section>
      )}

      {isEntry && (
        <Section title={tr('文件头代码', 'File preamble')}>
          <CodeArea
            value={project.preamble ?? ''}
            rows={3}
            placeholder={tr('import 语句、常量、辅助函数……', 'imports, constants, helper functions…')}
            onCommit={(v) => setText('preamble', v)}
          />
          <p className="mt-1 text-[11px] leading-relaxed text-muted">
            {tr('写在导出文件开头（torch 与 nn 的导入之后），代码块和保留的类可以使用其中的名称。', 'Written at the top of the exported file (after the torch and nn imports); code blocks and kept classes can use its names.')}
          </p>
        </Section>
      )}

      <Section title={tr('提示', 'Tips')}>
        <p className="text-[12px] leading-relaxed text-muted">
          {tr(
            '选中一个节点查看和修改参数。拖动输出端口到另一个节点的输入端口即可连线。框选多个节点后按 Ctrl+G 打包为模块，双击模块节点进入。',
            'Select a node to see and edit its parameters. Drag from an output port to an input port to connect. Box-select nodes and press Ctrl+G to pack them into a module; double-click a module node to open it.',
          )}
        </p>
      </Section>
    </>
  );
}
