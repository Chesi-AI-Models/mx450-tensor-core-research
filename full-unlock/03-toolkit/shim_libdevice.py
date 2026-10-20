"""Compatibility shim: triton.language.extra.libdevice for triton 2.x

triton 3.x moved the libdevice bindings to triton.language.extra.libdevice.
triton 2.3.x keeps them in triton.language.math (200 names) and, partially,
in triton.language.extra.cuda.  This module re-exports the 2.x names under the
3.x path so that code written against triton 3.x can at least import.

Reference: xformers/importing.py -> libdevice_find() probes exactly these paths.
"""
import triton.language.math as _math
import triton.language.extra.cuda as _cuda

_NAMES = set()
for _mod in (_math, _cuda):
    for _n in dir(_mod):
        if _n.startswith("_"):
            continue
        try:
            globals()[_n] = getattr(_mod, _n)
            _NAMES.add(_n)
        except Exception:
            pass

__all__ = sorted(_NAMES)
