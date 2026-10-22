# Complete technical teardown of CMPUnlocker

> Upstream: `d3dx9/cmpunlocker` (GPL v2). **Its source is not distributed with this repository**; fetch it via
> `refs/refs-fetch-manifest.sh`.
> Target: NVIDIA CMP 170HX (GA100), 8 GB → 64 GB, restoring A100 compute
> Purpose of this document: understand every step of it, and judge which parts apply to the MX450

---

## 1. The problem it solves

**The CMP 170HX is a physically complete GA100 (the same silicon as the A100), locked into a mining card by OTP fuses +
firmware.**

| Item | Factory | After unlock |
|---|---|---|
| VRAM | 8 GB (the 10 GB variant is 40 GB) | 64 GB HBM2e (5 × 16 GB) |
| SM compute | limited by PLM | close to A100 |
| PCIe | Gen1 ×4 | Gen2 (best-effort) |

**Key premise: the HBM2e packages are physically 16 GB × 5 stacks; they are merely strap-limited to 2 GB per stack.**

---

## 2. The seven-step flow

```
1. The Falcon BootROM loads the contents of the .fwsignature_ga100 section into DMEM 0x800
   BEFORE verifying the signature
   ↑ that is the vulnerability itself
2. Replace that section's contents with a 24-DWORD ROP chain
3. The ROP chain executes: BAR0[write_addr] = write_value
4. Repeat 4 times, opening 4 PLM registers
5. With PLM open, the host driver writes CFG1/LMR (memory) + SS0/SS1 (compute)
6. Restore the original GSP signature; the driver cannot detect tampering
7. The driver initialises normally, with full memory + full SM clock
```

**The unlock is volatile (lost on power-off) and is replayed every second by a systemd daemon.**

---

## 3. The structure of the ROP chain (`tools/exploit_simulator.py`)

24 DWORDs, two of whose slots are filled at runtime:

```python
ROP_GADGETS = [
    ("header",      0x1100, 0x00000007),   # signature header
    ("canary_1",    0x5b40, 0xc0deca7e),
    ("write_value", 0xf754, None),         # ← filled at runtime
    ("canary_2",    0xf758, 0xc0deca7e),
    ("gadget_1",    0xf75c, 0x00000cbd),   # Falcon instruction
    ("write_addr",  0xf76c, None),         # ← filled at runtime
    ...
]
```

**Core: the effect of the whole chain is a single `BAR0[write_addr] = write_value`.**

The gadget values are Falcon instruction encodings (mined from existing code in the firmware's DMEM),
so **gadget addresses are tightly bound to the firmware version and cannot be ported across versions or architectures.**

---

## 4. The PLM target registers

```python
PLM_TABLE = [
    {"addr": 0x001FA7CC, "value": 0xFFFFf0ff, "name": "WPR_CFG"},
    {"addr": 0x009A0148, "value": 0xFFFFFFFF, "name": "FBPA"},
    {"addr": 0x001FA7C4, "value": 0xFFFFFFFF, "name": "WPR"},
    {"addr": 0x00823804, "value": 0xFFFFFFFF, "name": "FEAT"},
]
POST_PLM_WRITES = [
    {"addr": 0x0082381C, "value": 0x88888888, "name": "SS0 (SM speed)"},
    {"addr": 0x00823820, "value": 0x00000008, "name": "SS1"},
    {"addr": 0x009A0204, "value": 0x02669000, "name": "CFG1 (memory geometry)"},
    {"addr": 0x00100CE0, "value": 0x0000028A, "name": "LMR (memory rank)"},
]
```

**`0x0082381C` (SS0) is exactly the SM throttle register — in stage 2 this project did a little-endian search for all
eight of these addresses across Turing's GSP firmware, and the result was: all eight zero-hit.**

---

## 5. Can it be used on the MX450? Judged item by item

| Stage | Applicable to TU117? | Reason |
|---|---|---|
| the `.fwsignature` section exists | ✅ | this machine's `gsp_tu10x.bin` has `.fwsignature_tu11x` (4096 bytes) |
| BootROM loads before verifying | ❓ unknown | requires measurement; Turing's BootROM may differ |
| ROP gadget addresses | ❌ | gadgets are Falcon instruction encodings, tightly bound to the firmware version |
| **PLM registers** | ❌ | **Turing has no PLM mechanism** (all 8 addresses zero-hit) |
| **CFG1/LMR memory unlock** | ❌ | **the MX450 physically has only 2 GB; there is no capacity to unlock** |
| **SS0/SS1 compute unlock** | ❌ | **TU117 has no TC, so there is nothing to unlock** |

**Conclusion: CMPUnlocker's methodology (ROP injection to bypass signing) "may" be feasible on Turing,
but the targets it aims to unlock (PLM, VRAM capacity, SM compute) all do not exist on the MX450.**

**The only potentially valuable use: the ROP chain runs in HS mode on the Falcon and can therefore access the
`0x20000` segment (GR internal registers) that the host cannot see. This is the only technical path to break the
6-bit `HOST_IO_INDEX` limit.**

But it requires: ① measuring whether Turing's BootROM has the same bug; ② re-mining gadgets in Turing's DMEM;
③ constructing a new chain. The effort is measured in weeks, and it is a high-risk operation (modifying firmware +
injecting code).

---

## 6. Three things learned from it

1. **The premise of "unlocking" is that "something has been sealed".** The CMP 170HX has 80 GB of physical VRAM; the
   MX450 has 2 GB.
2. **The bypass point of signature verification is "timing", not "cryptography".** The BootROM loads first and
   verifies afterwards; that ordering is the vulnerability.
3. **ROP gadgets are not portable.** They are instruction fragments mined out of a specific firmware's DMEM.

---

## 7. License

Upstream `cmpunlocker` is **GPL v2**. This repository **no longer contains its source**, only this file (original
analysis, MIT).
The rest of this project is original and is not subject to the GPL by virtue of referencing that directory.
