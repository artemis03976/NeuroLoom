# Import corpus

Model files in the styles found in real repositories, written for NeuroLoom's
import tests (they are not copies of third-party code). Each file's header gives
the input shapes (`# inputs:`) and, when the entry is not the last unreferenced
class, the entry class (`# class:`).

`packages/core/test/corpus.test.ts` imports every file statically and
dynamically, checks that the generated code is equivalent to the original
(`state_dict` loads with `strict=True`, same outputs), and prints how much of
each file was structured.
