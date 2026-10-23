import { useState } from 'react';
import { tr } from '../i18n';
import type { Prompt } from '../ui';

/** One text field with validation, e.g. the class name for "pack as module". */
export function PromptDialog({ prompt, onClose }: { prompt: Prompt; onClose: () => void }) {
  const [value, setValue] = useState(prompt.initial);
  const error = prompt.validate(value.trim());
  const submit = () => {
    if (error) return;
    onClose();
    prompt.submit(value.trim());
  };
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/40 p-6" onMouseDown={onClose}>
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby="prompt-title"
        className="w-[min(420px,100%)] rounded-xl border border-line bg-surface shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <header className="border-b border-line px-5 py-3">
          <h2 id="prompt-title" className="text-[15px] font-semibold">
            {prompt.title}
          </h2>
        </header>
        <div className="px-5 py-4">
          <label className="flex flex-col gap-1 text-[12px] text-muted">
            {prompt.label}
            <input
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onFocus={(e) => e.target.select()}
              onKeyDown={(e) => e.key === 'Escape' && onClose()}
              spellCheck={false}
              className={`rounded-md border bg-bg px-2 py-1.5 font-mono text-[13px] text-fg outline-none focus:border-primary ${error ? 'border-error' : 'border-line'}`}
            />
          </label>
          {error && <p className="mt-1.5 text-[11.5px] text-error">{error}</p>}
        </div>
        <footer className="flex justify-end gap-2 border-t border-line px-5 py-3">
          <button type="button" className="rounded-md border border-line px-3 py-1.5 text-[13px] hover:bg-surface-2" onClick={onClose}>
            {tr('取消', 'Cancel')}
          </button>
          <button
            type="submit"
            disabled={!!error}
            className="rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-fg hover:bg-primary-hover disabled:opacity-50"
          >
            {tr('确定', 'OK')}
          </button>
        </footer>
      </form>
    </div>
  );
}
