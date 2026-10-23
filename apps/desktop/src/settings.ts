import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export interface Settings {
  pythonPath?: string;
  recent: string[];
  /** Interface language, as last chosen in the editor. */
  locale?: 'zh' | 'en';
}

const MAX_RECENT = 10;

/** Writes through a temp file and a rename, so a crash never leaves a half-written file. */
export function writeAtomic(path: string, text: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.tmp`;
  writeFileSync(tmp, text, 'utf8');
  try {
    renameSync(tmp, path);
  } catch (err) {
    rmSync(tmp, { force: true });
    throw err;
  }
}

export class SettingsStore {
  private readonly file: string;
  data: Settings;

  constructor(userDataDir: string) {
    this.file = join(userDataDir, 'settings.json');
    try {
      this.data = { recent: [], ...JSON.parse(readFileSync(this.file, 'utf8')) };
    } catch {
      this.data = { recent: [] };
    }
  }

  save(): void {
    writeAtomic(this.file, `${JSON.stringify(this.data, null, 2)}\n`);
  }

  addRecent(path: string): void {
    this.data.recent = [path, ...this.data.recent.filter((p) => p !== path)].slice(0, MAX_RECENT);
    this.save();
  }

  recent(): string[] {
    return this.data.recent.filter((p) => existsSync(p));
  }
}
