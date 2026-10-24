import { useEffect } from 'react';
import { tr } from '../i18n';
import { useCurrentModule } from '../store';
import { useReplay } from './replay';

/** One step lasts this long at 1×. */
const STEP_MS = 900;
const SPEEDS = [0.5, 1, 2, 4];

/** Controls of the forward replay (PRD F3.5), over the bottom of the canvas. */
export function ReplayBar() {
  const { open, playing, speed, enter, index, steps, stack } = useReplay();
  const replay = useReplay.getState();
  const mod = useCurrentModule();

  useEffect(() => {
    if (!open || !playing) return;
    const timer = setInterval(() => {
      if (!useReplay.getState().advance()) clearInterval(timer);
    }, STEP_MS / speed);
    return () => clearInterval(timer);
  }, [open, playing, speed]);

  // An edit changes the steps; the replay carries on from where it was.
  useEffect(() => useReplay.getState().refresh(), [mod]);

  if (!open) return null;
  const finished = index >= steps.length;
  const button = 'grid h-7 min-w-7 place-items-center rounded-md px-1.5 text-[13px] hover:bg-surface-2 disabled:opacity-40';
  return (
    <div
      role="toolbar"
      aria-label={tr('前向回放', 'Forward replay')}
      className="pointer-events-auto absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 whitespace-nowrap rounded-xl border border-line bg-surface px-2 py-1.5 text-[12px] shadow-lg"
    >
      <span className="px-1.5 font-medium">{tr('前向回放', 'Forward replay')}</span>
      <button type="button" className={button} title={tr('从头开始', 'Restart')} aria-label={tr('从头开始', 'Restart')} onClick={() => replay.restart()}>
        ⏮
      </button>
      <button
        type="button"
        className={`${button} bg-primary text-primary-fg hover:bg-primary-hover`}
        title={playing ? tr('暂停', 'Pause') : tr('播放', 'Play')}
        aria-label={playing ? tr('暂停', 'Pause') : tr('播放', 'Play')}
        onClick={() => replay.setPlaying(!playing)}
      >
        {playing ? '❚❚' : '▶'}
      </button>
      <button
        type="button"
        className={button}
        disabled={finished && stack.length === 0}
        title={tr('单步', 'Step')}
        aria-label={tr('单步', 'Step')}
        onClick={() => {
          replay.setPlaying(false);
          replay.advance();
        }}
      >
        ⏭
      </button>
      <span className="w-20 text-center font-mono text-[11.5px] text-muted">
        {finished ? tr('完成', 'done') : `${Math.max(0, index + 1)} / ${steps.length}`}
        {stack.length > 0 && ` · ${tr('第', 'depth ')}${stack.length}${tr(' 层', '')}`}
      </span>
      <select
        value={speed}
        onChange={(e) => replay.setSpeed(Number(e.target.value))}
        aria-label={tr('速度', 'Speed')}
        className="rounded-md border border-line bg-bg px-1 py-0.5 font-mono text-[11.5px] outline-none"
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
      <label className="ml-1 flex items-center gap-1 text-muted" title={tr('遇到模块节点时进入内部播放，否则模块节点整体算一步', 'Play inside module nodes instead of taking each as one step')}>
        <input type="checkbox" checked={enter} onChange={(e) => replay.setEnter(e.target.checked)} />
        {tr('进入模块', 'Enter modules')}
      </label>
      <button type="button" className={`${button} ml-1`} title={tr('关闭回放', 'Close the replay')} aria-label={tr('关闭回放', 'Close the replay')} onClick={() => replay.close()}>
        ✕
      </button>
    </div>
  );
}
