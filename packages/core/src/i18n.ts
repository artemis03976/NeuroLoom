// Locale for user-facing messages produced by the core (diagnostics, inference
// errors). The editor sets it and re-runs analysis when the language changes.

export type Locale = 'zh' | 'en';

let current: Locale = 'zh';

export function setLocale(locale: Locale): void {
  current = locale;
}

export function getLocale(): Locale {
  return current;
}

/** A message in the current locale. */
export const msg = (zh: string, en: string): string => (current === 'en' ? en : zh);
