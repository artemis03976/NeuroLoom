// A small Python tokenizer for the read-only code preview. Monaco replaces it in M1.
export interface Token {
  text: string;
  kind?: 'kw' | 'str' | 'num' | 'com' | 'fn' | 'self';
}

const PATTERN =
  /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(import|from|as|class|def|return|super|None|True|False|for|in|if|else|pass|tuple)\b|\b(self)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;

export function tokenizeLine(line: string): Token[] {
  const tokens: Token[] = [];
  let last = 0;
  for (const m of line.matchAll(PATTERN)) {
    if (m.index > last) tokens.push({ text: line.slice(last, m.index) });
    const kind = m[1] ? 'com' : m[2] ? 'str' : m[3] ? 'kw' : m[4] ? 'self' : m[5] ? 'num' : 'fn';
    tokens.push({ text: m[0], kind });
    last = m.index + m[0].length;
  }
  if (last < line.length) tokens.push({ text: line.slice(last) });
  return tokens;
}
