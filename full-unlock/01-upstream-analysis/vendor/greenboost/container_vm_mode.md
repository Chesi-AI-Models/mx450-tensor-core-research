# GreenBoost v2.5 ; Blended Shim: Three-Path Overflow Architecture
---

Author: Jerry Nguyen

Contributor: Ferran Duarri


## Overview

The GreenBoost CUDA shim (`libgreenboost_cuda.so`) can route VRAM overflow
allocations through three distinct paths, selected automatically at runtime:

| Path | Name | Requires kernel module | Best for |
|------|------|----------------------|----------|
| **A** | DMA-BUF + kernel | Yes — `greenboost.ko` | Bare-metal Linux (default) |
| **B** | HostReg no-kernel | No | Containers, VMs, Docker, LXC, KVM |
| **C** | UVM / cuMemAllocManaged | No (nvidia_uvm.ko only) | Universal last resort |

The shim tries Path A first.  If `/dev/greenboost` is unavailable (no kernel
module loaded), it falls through to Path B automatically.  Path C is the
final fallback if Path B fails or is explicitly disabled.

---

## What was preserved from Jerry Nguyen (MR !3)

Jerry Nguyen's merge request introduced a **kernel-module-free** approach to
VRAM overflow using the standard CUDA driver API:

1. `mmap(MAP_ANONYMOUS)` to allocate anonymous system RAM pages.
2. `cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_DEVICEMAP)` to pin those
   pages and make them GPU-accessible without any kernel driver involvement.
3. `cuMemHostGetDevicePointer(&dev_ptr, ptr, 0)` to obtain a CUDA device
   pointer the application can use normally.

This is the complete core of Path B.  No changes were made to this mechanism —
it works exactly as Jerry proposed.  The key insight from MR !3 is that the
CUDA driver itself, via `cuMemHostRegister`, can pin host memory and expose it
to the GPU without a custom kernel module, making the overflow approach viable
inside containers and virtual machines where `greenboost.ko` cannot be loaded.

---

## What GreenBoost v2.5 features were added to Path B

The following v2.5 capabilities were integrated into Jerry's base approach:

### 2 MB hugepage preference (new)
Before falling back to 4 K pages, Path B attempts `mmap` with
`MAP_HUGETLB | MAP_HUGE_2MB`.  This reduces TLB pressure for large model
allocations (tens of gigabytes) and can improve PCIe DMA throughput by ~10–15%
on kernel ≥ 5.14 with `vm.nr_hugepages` available.

### Hash-table tracking (inherited from v2.5 shim)
Every Path B allocation is recorded in the same open-addressed 131 072-slot
hash map used by Path A and Path C.  This ensures that `cuMemFree_v2` /
`cudaFree` correctly unregisters the host memory with `cuMemHostUnregister`
and calls `munmap` — no leak even for apps that free allocations out of order.

### Transparent fallback ordering (new)
Path B was inserted *between* Path A and Path C in `gb_overflow_alloc()`.  The
calling code (`cuMemAlloc_v2`, `cuMemAllocAsync`, `cudaMalloc`, etc.) sees a
single function call and is unaware which path succeeded.  If Path A fails
(no `/dev/greenboost`) and Path B also fails (CUDA driver rejects the
registration), the shim proceeds to Path C (UVM) rather than returning OOM
immediately.

### `GREENBOOST_NO_HOSTREG=1` escape hatch (new)
Users who want to force UVM-only mode (e.g., for profiling or debugging) can
set `GREENBOOST_NO_HOSTREG=1` to skip Path B entirely.

### Debug banner Path A/B/C labels (new)
`GREENBOOST_DEBUG=1` now prints the status of all three paths at startup:
```
[GreenBoost] Path A (DMA-BUF+kernel) : enabled — mmap+GB_IOCTL_PIN_USER_PTR+HostReg (best bandwidth)
[GreenBoost] Path B (HostReg/no-kmod): available — mmap+cuMemHostRegister (containers/VMs, no greenboost.ko needed)
[GreenBoost] Path C (UVM/managed)    : available — cuMemAllocManaged+cuMemAdvise (last resort)
```

### All v2.5 shim features remain active on Path B
- Two-stage `RTLD_NOLOAD` constructor guard (safe system-wide injection via `/etc/ld.so.preload`)
- `dlsym` hook intercepting Ollama's `dlopen+dlsym` GPU API lookups
- `dlopen` hook stripping `RTLD_DEEPBIND`
- Virtual VRAM inflation hooks (`cuDeviceTotalMem_v2`, `nvmlDeviceGetMemoryInfo`)
- `cuMemGetInfo` hook (reports physical VRAM + system RAM pool to CUDA)
- `GREENBOOST_ACTIVE` opt-in for lazy-dlopen apps (Ollama, vLLM, PyTorch)
- Async prefetch worker thread with `MADV_WILLNEED`
- Async allocation hooks (`cuMemAllocAsync`, `cudaMallocAsync`)
- `gb_atoll()` GLIBC_2.38 snap compatibility fix

---

## When to use Path A (DMA-BUF + kernel module) — the standard

**Use Path A (the default) on bare-metal Linux.**

Path A is the recommended configuration for all standard installations.
It provides the best possible bandwidth and the most robust integration:

- **Pinned hugepages via kernel buddy allocator.** `greenboost.ko` allocates
  2 MB compound pages from the kernel's buddy allocator, which are physically
  contiguous and guaranteed not to be swapped or migrated by the kernel pager.
  This gives the GPU DMA engine the best possible access pattern over PCIe.

- **Kernel-controlled memory pressure watchdog.** The kernel module runs a
  kthread that monitors system RAM and NVMe swap pressure.  If the system is
  running low, it signals userspace via eventfd before things become dangerous
  — something a pure-userspace path cannot do.

- **T3 NVMe tier.** The kernel module manages the Tier 3 NVMe overflow pool
  (`/swap_nvme.img`), which can extend the effective memory to hundreds of
  gigabytes.  Path B has no T3 equivalent — when System DDR is exhausted it falls
  back to Path C (UVM paging to swap), which is slower and less controlled.

- **`/sys/class/greenboost/greenboost/pool_info` live monitoring.** The sysfs
  interface exposed by the kernel module lets you monitor T2 allocation in real
  time.  `watch -n1 'cat /sys/class/greenboost/greenboost/pool_info'` gives
  instant visibility into how much System DDR the model is consuming.

**Summary: on bare metal, always run `sudo ./greenboost_setup.sh full-install`
and use Path A.  Path B exists for environments where Path A is impossible.**

---

## When to use Path B (no-kernel, containers/VMs)

Use Path B when you cannot load a custom kernel module:

### Docker / Podman containers
Docker containers run in a shared kernel namespace.  You cannot `insmod` a
custom module inside a container — the host kernel does not allow it (and the
module ABI may differ).  Path B lets GreenBoost work inside CUDA-enabled
Docker images without any host-side kernel change beyond the NVIDIA Container
Toolkit being present.

```bash
# Inside a Docker container — Path A is impossible, Path B activates automatically
docker run --gpus all \
  -e GREENBOOST_ACTIVE=1 \
  -v /usr/local/lib/libgreenboost_cuda.so:/usr/local/lib/libgreenboost_cuda.so \
  -e LD_PRELOAD=/usr/local/lib/libgreenboost_cuda.so \
  my-llm-image python run_model.py
```

### LXC / LXD unprivileged containers
Same constraint as Docker — custom kernel modules cannot be loaded from within
an unprivileged container.

### KVM / QEMU virtual machines with GPU passthrough
In a KVM guest with PCI passthrough of an NVIDIA GPU, the guest kernel can
load `nvidia.ko` and `nvidia-uvm.ko` (the NVIDIA driver), but `greenboost.ko`
would need to be built against the *guest* kernel — a separate build step that
many users want to avoid.  Path B provides VRAM overflow without needing to
build and load a guest kernel module.

### WSL2 (Windows Subsystem for Linux 2)
WSL2 runs a Microsoft-patched Linux kernel that does not support arbitrary
third-party kernel modules via `insmod`.  The NVIDIA CUDA driver for WSL2 is
provided by the host Windows driver and exposed through `/dev/dxg`.  Path B's
`cuMemHostRegister` path works correctly on WSL2 CUDA because it only calls
into the CUDA driver API.

### Shared HPC / cloud GPU instances
On shared HPC clusters or cloud VM instances (AWS, GCP, Azure GPU VMs) you
typically do not have root access to load kernel modules.  Path B allows
GreenBoost's VRAM overflow to work in `srun`/`sbatch` jobs without any
sysadmin involvement.

---

## Path B limitations compared to Path A

| Feature | Path A (DMA-BUF) | Path B (HostReg) |
|---------|-----------------|-----------------|
| Hugepages | Kernel-allocated, guaranteed 2 MB | Best-effort `MAP_HUGETLB` (may fall back to 4 K) |
| T3 NVMe tier | Yes — kernel watchdog + swap subsystem | No — System DDR only, then UVM paging |
| Live sysfs monitoring | `/sys/class/greenboost/pool_info` | Not available |
| Memory pressure watchdog | Kernel kthread, eventfd signal | Not available |
| Physical page pinning | Hard-pinned by kernel (cannot be evicted) | Soft-pinned by CUDA driver (may be re-registered) |
| PCIe bandwidth (typical) | ~50 GB/s (System DDR dual-channel, pinned hugepages) | ~32–45 GB/s (depends on page size and TLB miss rate) |
| Requires `greenboost.ko` | Yes | No |
| Works in Docker/VM/WSL2 | No | Yes |

---

## Environment variable reference

| Variable | Default | Effect |
|---|---|---|
| `GREENBOOST_USE_DMA_BUF` | `1` | `0` = skip Path A entirely (jump to B then C) |
| `GREENBOOST_NO_HOSTREG` | `0` | `1` = skip Path B (jump straight to Path C / UVM) |
| `GREENBOOST_ACTIVE` | unset | Set to `1` for lazy-dlopen apps (Ollama, vLLM, PyTorch) |
| `GREENBOOST_DEBUG` | `0` | `1` = verbose per-path status at startup + per-alloc log |
| `GREENBOOST_VRAM_HEADROOM_MB` | `2048` | Keep this many MB free in VRAM before overflowing |
| `GREENBOOST_VIRTUAL_VRAM_MB` | auto | Override reported virtual VRAM size |
| `GREENBOOST_CUDART_PATH` | auto | Explicit `libcudart.so` path |
| `GREENBOOST_DISABLE` | `0` | `1` = disable shim entirely for one process |

---

## License and attribution

GreenBoost is GPL v2.  Attribution to Ferran Duarri is required in all forks
and derivatives.

Path B was originally proposed by **Jerry Nguyen** (GitLab
[@jerry.nguyen](https://gitlab.com/jerry.nguyen)) in merge request
[!3](https://gitlab.com/IsolatedOctopi/nvidia_greenboost/-/merge_requests/3).
The `cuMemHostRegister(DEVICEMAP)` mechanism, the kernel-module-free concept,
and the container/VM use-case rationale all originate from Jerry's contribution.
The hugepage preference, hash-table integration, fallback ordering, env var
controls, and debug banner were added by Ferran Duarri when integrating MR !3
into the v2.5 blended shim.

```
Copyright (C) 2026 Ferran Duarri
Path B concept: Copyright (C) 2026 Jerry Nguyen
```
