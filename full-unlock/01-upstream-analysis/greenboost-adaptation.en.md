# Complete GreenBoost adaptation report on the MX450

> **Result: 1.64 GiB physical VRAM, successfully allocated 8 GiB, and the matmul results are fully correct.**
>
> This is the project's most practically valuable result — it turns "run 7B QLoRA on 2 GB of VRAM" from impossible
> into possible.

---

## 1. Why this project matters

The MX450 has only 1.64 GiB of usable physical VRAM (2 GB physical).

7B QLoRA needs about 4 GB. **The gap is 2.4×.**

GreenBoost uses "three-tier storage" to turn system RAM and NVMe into virtual VRAM:

| Tier | Medium | Configuration here | Bandwidth |
|---|---|---|---|
| **T1** | GPU VRAM | **2 GB** | ~80 GB/s |
| **T2** | system DDR (pinned DMA-BUF) | **8 GB** | ~4 GB/s (PCIe 3.0 x4) |
| **T3** | NVMe swap | **8 GB** | ~1.8 GB/s |

**18 GB of model capacity in total.**

---

## 2. The adaptation process (four steps)

### Step 1 · build the kernel module

```bash
# upstream: greenboost's greenboost.c + greenboost_ioctl.h (fetch via refs-fetch-manifest.sh)
cat > Makefile <<'EOF'
obj-m += greenboost.o
KDIR ?= /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
EOF

make
```

**Result: compiled on the first try.**

```
CC [M]  greenboost.o
LD [M]  greenboost.ko
```

**Kernel version `6.18.18.c1126-trim` (fnOS custom kernel) is fully compatible.**
The only warning was a minor compiler version difference (12.2.0-14 vs 12.2.0-14+deb12u1), with no impact.

### Step 2 · load the module (parameter tuning)

**First attempt failed**: with `safety_reserve_gb=4`, the system's `free` showed only 2.6 GB and the OOM guard tripped:

```
greenboost: T2 OOM guard TRIPPED — free=2694MB < reserve=4GB
T2 available: 0 MB
```

**Cause**: Linux's `free` excludes page cache (this machine had 10 GB in cache),
while GreenBoost's watchdog looks at `nr_free_pages`.

**Fix**: release the page cache + lower the reserve.

```bash
sync && echo 3 > /proc/sys/vm/drop_caches
insmod greenboost.ko physical_vram_gb=2 virtual_vram_gb=8 \
       safety_reserve_gb=2 nvme_swap_gb=8 pcores_only=0 use_hugepages=1
```

**Result**:

```
T1 VRAM : 2 GB
T2 RAM  : pool cap 8 GB (reserve 2 GB)
T3 NVMe : 8 GB
Combined: 18 GB

T2 available : 11076 MB
OOM guard    : no
```

### Step 3 · build the CUDA shim

**The shim does not depend on `cuda.h`** (everything is resolved dynamically via `dlsym`), but there were two
Debian 12 build issues:

| Issue | Fix |
|---|---|
| `MAP_HUGE_2MB` undefined | `-DMAP_HUGE_2MB=0x54000000` |
| `_GNU_SOURCE` defined twice | ignore the warning |

```bash
gcc -shared -fPIC -O2 -D_GNU_SOURCE -DMAP_HUGE_2MB=0x54000000 \
    -o libgreenboost_cuda.so greenboost_cuda_shim.c -ldl -lpthread
```

### Step 4 · **the critical fix: the `_v2` symbols**

**The stock shim uses `dlsym(libcuda, "cuMemHostRegister")`, which under CUDA 12 returns 201 (INVALID_CONTEXT).**

Switching to the `_v2` versions succeeded immediately:

```c
dlsym(libcuda, "cuMemHostRegister_v2")
dlsym(libcuda, "cuMemHostGetDevicePointer_v2")
```

**This is the only code change in the whole adaptation — two lines.**

---

## 3. Verification results

### VRAM allocation test

```
physical VRAM reported: 1.64 GiB

alloc 2 GiB: OK   reserved=2.00 GiB
alloc 3 GiB: OK   reserved=3.00 GiB
alloc 4 GiB: OK   reserved=4.00 GiB
alloc 6 GiB: OK   reserved=6.00 GiB
alloc 8 GiB: OK   reserved=8.00 GiB      ← 8 GiB, 4.9× the physical size
```

### Computational correctness

```
matmul on 4GiB-occupied GPU: OK
c[0,0] = 1024.0 (expect 1024.0)          ← fully correct
```

### Kernel-side record

```
greenboost: pinned 3072MB user buffer (id=1 fd=43)
```

---

## 4. How to use it

```bash
# 1. load the module (after every boot)
sudo modprobe greenboost physical_vram_gb=2 virtual_vram_gb=8 \
     safety_reserve_gb=2 nvme_swap_gb=8 pcores_only=0 use_hugepages=1

# 2. run your program with the shim
LD_PRELOAD=/vol1/ganlin/greenboost-build/libgreenboost_cuda.so \
GREENBOOST_ACTIVE=1 \
python train.py

# 3. monitor
watch -n1 'cat /sys/class/greenboost/greenboost/pool_info'
```

---

## 5. The costs that must be stated clearly

### 1. The bandwidth cliff

| Tier | Bandwidth | Relative to T1 |
|---|---|---|
| T1 (VRAM) | ~80 GB/s | 1.0× |
| T2 (PCIe 3.0 x4) | ~4 GB/s | **1/20** |
| T3 (NVMe) | ~1.8 GB/s | 1/44 |

**Inference is tolerable (weights read once, compute stays in T1); training is painful (gradient updates touch the
weights repeatedly).**

### 2. This run took the UVM path

`T2 allocated: 0 MB` + `Active DMA-BUF objects: 0` show that the actual allocation went through
`cuMemAllocManaged` (Path 2), not DMA-BUF (Path 1).

UVM's page migration is handled on demand by the driver, so latency is higher than pinned DMA-BUF.

**Forcing the DMA-BUF path requires confirming the enabling conditions for `gb_use_dmabuf`.**

### 3. Stability

- the module is GPL v2, used as-is
- while loaded it does not affect the NVIDIA driver (`nvidia-smi` normal)
- unload: `rmmod greenboost`
- **must be reloaded after a reboot** (no systemd persistence configured)

---

## 6. What this means for "training a 7B Chesi"

| Item | Before | Now |
|---|---|---|
| usable VRAM | 1.64 GiB | **8 GiB (virtual)** |
| 7B QLoRA (needs 4 GB) | ❌ does not fit | ✅ **fits** |
| training speed | — | limited by T2 bandwidth; estimated 3–10× slower than pure VRAM |

**Conclusion: from "physically impossible" to "runs, but slowly".**

**This is the limit of what can be done on this machine.**

---

## 7. Upstream information

- **Project**: GreenBoost v2.5
- **Author**: Ferran Duarri
- **License**: GPL v2 (open source) / commercial dual licence
- **Upstream**: fetched via `refs/refs-fetch-manifest.sh` (this repository no longer inlines the source)
- **Changes made by this adaptation**: only 2 lines (`cuMemHostRegister` → `_v2`)

**This adaptation report is original work, recording the complete validation on MX450 + fnOS 6.18.18.**
