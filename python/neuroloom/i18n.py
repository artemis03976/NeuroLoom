"""Language of user-facing messages, following the editor's (zh or en).

Messages are written next to where they are produced as t('中文', 'English')
and must be evaluated then, not stored in module-level constants, so that a
later set_locale() applies to them.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

LOCALES = ("zh", "en")

_locale = "zh"


def set_locale(locale: str) -> None:
    """Sets the message language; anything other than "zh" or "en" means Chinese."""
    global _locale
    _locale = locale if locale in LOCALES else "zh"


def get_locale() -> str:
    return _locale


@contextmanager
def using_locale(locale: str | None) -> Iterator[str]:
    """Messages produced inside the block use `locale` (the current one when None); yields the locale in effect."""
    previous = _locale
    if locale is not None:
        set_locale(locale)
    try:
        yield _locale
    finally:
        set_locale(previous)


def t(zh: str, en: str) -> str:
    """The message in the current language."""
    return en if _locale == "en" else zh
