// Interface language (PRD F10.2). Strings live next to where they are used as
// tr('中文', 'English'); the core's diagnostics follow the same locale.
import { getLocale, setLocale as setCoreLocale, type Locale } from '@neuroloom/core';
import { create } from 'zustand';

export type { Locale };

const KEY = 'neuroloom.locale';

function initialLocale(): Locale {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'zh' || saved === 'en') return saved;
  } catch {
    // Storage may be unavailable; fall back to the system language.
  }
  return typeof navigator !== 'undefined' && !navigator.language.toLowerCase().startsWith('zh') ? 'en' : 'zh';
}

setCoreLocale(initialLocale());

/** The text for the current language. */
export const tr = (zh: string, en: string): string => (getLocale() === 'en' ? en : zh);

interface LocaleState {
  locale: Locale;
  setLocale(locale: Locale): void;
}

export const useLocale = create<LocaleState>()((set) => ({
  locale: getLocale(),
  setLocale(locale) {
    setCoreLocale(locale);
    try {
      localStorage.setItem(KEY, locale);
    } catch {
      // Not persisted; the choice still applies to this session.
    }
    set({ locale });
  },
}));

/** Bilingual data (node names, summaries) in the current language. */
export const pick = (text: { zh: string; en: string }): string => (getLocale() === 'en' ? text.en : text.zh);
