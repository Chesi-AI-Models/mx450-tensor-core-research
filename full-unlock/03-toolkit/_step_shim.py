# -*- coding: utf-8 -*-
"""在 NAS 上：安装 libdevice shim 并逐步试探 FlagGems 能走到哪一步"""
import os, sys, shutil, traceback

SHIM_SRC = "/tmp/shim_libdevice.py"
SITE = "/vol1/ganlin/venv/lib/python3.11/site-packages"
TGT = os.path.join(SITE, "triton/language/extra/libdevice.py")

print("=" * 70)
print("STEP 1: install shim")
print("=" * 70)
print("shim exists at /tmp:", os.path.exists(SHIM_SRC))
if os.path.exists(SHIM_SRC):
    try:
        shutil.copyfile(SHIM_SRC, TGT)
        print("copied ->", TGT)
    except PermissionError as e:
        print("PermissionError:", e)
        rc = os.system("sudo -n cp %s %s" % (SHIM_SRC, TGT))
        print("os.system rc =", rc)
print("target exists:", os.path.exists(TGT), "size:", os.path.getsize(TGT) if os.path.exists(TGT) else 0)

print()
print("=" * 70)
print("STEP 2: import triton.language.extra.libdevice")
print("=" * 70)
try:
    from triton.language.extra import libdevice as d
    print("OK names =", len(getattr(d, "__all__", [])))
    for f in ("exp", "pow", "log", "sqrt", "tanh", "erf", "rsqrt", "fast_expf"):
        print("   %-10s %s" % (f, hasattr(d, f)))
except Exception:
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("STEP 3: import flag_gems")
print("=" * 70)
try:
    import flag_gems
    print("OK flag_gems", flag_gems.__version__)
except Exception:
    traceback.print_exc()
    print()
    print(">>> import failed, stop here.")
    sys.exit(2)

print()
print("=" * 70)
print("STEP 4: backend state")
print("=" * 70)
try:
    from flag_gems.runtime import backend as be
    st = be._state
    for a in ("vendor_name", "device_name", "has_arch", "arch"):
        if hasattr(st, a):
            print("   %-12s = %s" % (a, getattr(st, a)))
    try:
        am = getattr(st.vendor_module, "ARCH_MAP", None)
        print("   ARCH_MAP keys =", list(am.keys()) if am else am)
    except Exception as e:
        print("   ARCH_MAP err:", e)
except Exception:
    traceback.print_exc()
print()
print(">>> shim worked far enough to import flag_gems.")
