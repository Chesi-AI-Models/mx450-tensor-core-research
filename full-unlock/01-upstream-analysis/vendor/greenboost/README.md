# GreenBoost : 3-Tier GPU Memory Extension for Linux

**Author:** Ferran Duarri
**License:** GPL v2 (open-source)
**Version:** 2.5

<div align="center">

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/nvidia_greenboost)

</div>

---

I have been quietly working on this for a while and figured it was time to share it with the
community. This is a Linux kernel module + CUDA userspace shim that transparently extends
GPU VRAM using system DDR RAM and NVMe storage, so you can run large language models that
exceed your GPU memory without modifying the inference software at all.

**GreenBoost does not replace or modify the official NVIDIA kernel drivers** (`nvidia.ko`,
`nvidia-uvm.ko`). It loads as a completely independent kernel module alongside them and works
at the CUDA allocation layer.

---

## The problem I was trying to solve

I have an RTX 5070 with 12 GB VRAM and I wanted to run `glm-4.7-flash:q8_0`, a 31.8 GB
model. The standard options are:

- **Offload layers to CPU** ; works, but drops tok/s significantly and completely breaks on
  long contexts. More on this below.
- **Use a smaller quantization** ; you lose quality. At q4_0 the model is noticeably worse
  on reasoning tasks.
- **Buy a bigger GPU** ; a 48 GB card costs more than a complete workstation.

None of those felt right, so I built an alternative: route the overflow memory to system DDR
via DMA-BUF, which gives the GPU direct access to system RAM over the PCIe link without any
CPU copy involved.

My system: i9-14900KF, RTX 5070 12 GB (Blackwell, compute capability 12.0), 64 GB
DDR4-3600, Samsung 990 EVO Plus 4 TB NVMe, Ubuntu 26.04, kernel 6.19, NVIDIA driver 580.x.

---

## Why not ik_llama.cpp / CPU offload?

I really like the concept of the CPU taking care of tensor compute. The idea is elegant.
The problem is the compute power is not there, and the situation gets much worse the moment
you work with long contexts.

The thing that kills ik_llama.cpp for my use case is **prefill**. I am running 198K context.
When you throw a long document or a multi-turn conversation at the model, the prefill stage
processes potentially thousands of tokens in parallel batches, and that is where GPU compute
absolutely demolishes CPU.

With ik_llama.cpp, those CPU-side layers have to run matrix multiplications on the full batch
using AVX vector instructions. The i9-14900KF is a powerful CPU, but it is not built for this. 
With GreenBoost, the GPU handles the entire model during prefill, streaming the System DDR
layers over PCIe while tensor cores do the work in parallel. Prefill is easily 5–10× faster.

If you are running short prompts, ik_llama.cpp could be a reasonable solution. But with 198K context and real workloads, slow prefill
absolutely kills the experience. With an autonomous workflow where the LLM creates plans and
implements them across long multi-turn sessions, how are you supposed to achieve that with
ik_llama.cpp? It will collapse under the context length before it gets anywhere useful, and that without even considering the compatibility issues you face with ik_llama.cpp on different setups.

There is no reason to use llama.cpp and/or ik_llama.cpp with greenboost; the only useful scenario would be if you have an AMD GPU that cannot benefit from greenboost.

**The deeper difference: system-level vs application-level.**

ik_llama.cpp is an application-level solution. It teaches itself to spill weights into system
RAM when VRAM fills up, but that is entirely internal to the llama.cpp process. From the
operating system's perspective, your GPU still has 12 GB. PyTorch, ComfyUI, any other app
that calls `cudaMalloc` hits the same 12 GB wall as always. They have no visibility into what
llama.cpp is doing internally.

GreenBoost works at a fundamentally different level. The kernel module carves out a large pool
of pinned DDR RAM and makes it GPU-accessible via DMA-BUF, a standard Linux mechanism for
sharing memory between devices. The CUDA shim intercepts memory allocation calls before they
reach the real GPU driver. Any process running with that shim; PyTorch, ComfyUI, Ollama,
anything, transparently gets access to that expanded pool. No per-application changes needed.

**GreenBoost is not a cache.** A common misconception: people hear "overflow to System DDR" and
assume there is a hit/miss/eviction policy somewhere. There is not. A 31 GB model just
*lives* in T1+T2 (VRAM + System DDR pool). The GPU addresses everything directly via PCIe DMA The GPU addresses everything directly via PCIe DMA (Direct Memory Access).
The model weights simply live in VRAM and System DDR at the same time, permanently. The GPU reads T1 weights at full VRAM speed and T2 weights over the PCIe link at ~32 GB/s PCIe 4 / ~64 GB/s PCIe 5. 
T2 is slower because of the hardware limitations itself between the GPU and the RAM, not because of any software caching mechanism.

---

## How it works

Two components cooperate:

**1. Kernel module (`greenboost.ko`)**

Allocates pinned DDR pages using the buddy allocator (2 MB compound pages for efficiency)
and exports them as DMA-BUF file descriptors. The GPU imports these pages as CUDA external
memory via `cudaImportExternalMemory`. From CUDA's perspective, those pages look like
device-accessible memory, it does not know they live in system RAM. The PCIe link handles
the actual data movement (~32 GB/s on PCIe 4.0 x16, ~64 GB/s on PCIe 5.0 x16).

A sysfs interface (`/sys/class/greenboost/greenboost/pool_info`) lets you monitor usage
live. A watchdog kernel thread monitors RAM and NVMe pressure and signals userspace before
things get dangerous.

**2. CUDA shim (`libgreenboost_cuda.so`, injected via `LD_PRELOAD`)**

Intercepts `cudaMalloc`, `cudaMallocAsync`, `cuMemAllocAsync`, `cudaFree`, and `cuMemFree`.
Small allocations pass straight through to the real CUDA runtime. Large ones; KV cache,
model weights overflowing VRAM, are redirected to the kernel module and imported back as
CUDA device pointers.

One tricky part worth mentioning: Ollama resolves GPU symbols via `dlopen` + `dlsym`
internally, which bypasses `LD_PRELOAD` entirely for those symbols. To handle this, the shim
also intercepts `dlsym` itself and returns hooked versions of `cuDeviceTotalMem_v2` and
`nvmlDeviceGetMemoryInfo`. Without this, Ollama sees only 12 GB and puts layers on the CPU.

---

## Memory tiers

| Tier | Device | Capacity | Bandwidth | Used for |
|------|--------|----------|-----------|---------|
| T1 | RTX 5070 VRAM | 12 GB | ~336 GB/s | Hot layers, active compute | << GPU I've been building GreenBoost on
| T2 | System DDR pool | 51 GB | ~32 GB/s (PCIe 4.0 x16, ~64 GB/s on PCIe 5.0 x16) | KV cache, cold weights |
| T3 | NVMe swap | 64 GB | ~1.8 GB/s | Safety overflow (rarely hit) |

T1 holds the hot layers. T2 holds the rest of the model and the KV cache overflow at long
contexts. T3 is a safety net, the model and KV cache at normal working context lengths fit
comfortably in T1+T2, and T3 should rarely be touched in normal operation.

**Why GPU monitoring tools show ~65% VRAM used, not 100%**

This is expected and correct. When you run a model with GreenBoost, the shim tells Ollama
that the total GPU memory is 63 GB (T1 + T2 combined). Ollama then distributes the model
layers across all 63 GB. The layers that happen to land in the first 12 GB end up in real
VRAM; the rest land in the System DDR pool.
** 63Gb is an example based on my workstation hardware.

What `nvidia-smi` or any GPU monitor shows is only the physical VRAM, it has no visibility
into the DDR pool. So if you are running a 32 GB model and the monitor shows 7–8 GB of VRAM
used, that does not mean the model is only partially loaded. It means 7–8 GB of it is in
VRAM and the remaining ~24 GB is in System DDR, accessed by the GPU directly over PCIe.

On top of that, GreenBoost reserves headroom in VRAM for two things that must stay fast:
the KV cache (rewritten on every generated token) and the activation buffers (temporary
workspace the GPU uses during each layer's computation). Filling VRAM to 100% with weights
would leave no room for these, and generation would fail or spill to System DDR in the
worst possible way; random, uncontrolled, under pressure.

So the ~65% VRAM figure is not waste. It is: model layers that fit in VRAM + reserved space
for KV cache + activation headroom. The rest of the model is in System DDR doing exactly
what it is supposed to do.


## What this code is based on, and where it comes from

I started coding at 12y old, got a JavaScript book from a local store. A few years later I was into Cisco networking basics,
capturing TCP/IP packets, playing with Linux back when Ubuntu was just starting to get traction.

More recently I had been modifying VMware kernel modules to get them running on newer
Linux kernels. That gave me a working understanding of how kernel modules are structured,
how they interact with the rest of the system.

No that long ago I took neural network and machine learning lessons, trained a couple of
recognition models, and got far enough into the TensorFlow Developer certification path
that I was essentially ready to sit the exam. I got into a relationship, live changed, less control of the time, 
and I ended up missing the certification exam. That one still stings a bit.

I use AI inference every day at work. Watching tokens burn through cloud APIs, knowing
I had a GPU doing nothing useful, it eventually clicked. The question I started asking was simple: 
can I run a model like `glm-4.7-flash` properly on my own machine, with the GPU tensor cores doing all the compute, 
without CPU spill?

That was the spark for GreenBoost to materialize.

The first thing I needed to understand was whether NVIDIA's tensor cores could handle
the full model if I found a way to give the GPU access to more memory. 

That led me to NVIDIA's open-source kernel modules source, the code published on GitHub. 

The NVIDIA kernel modules handle the low-level hardware side memory management, 
unified virtual memory, DMA. CUDA sits on top of that as the runtime and API layer:
it's what applications actually talk to when they want to allocate memory, move data, or run compute on the GPU. Tensor cores computations, transfers, allocations, all of it flows through CUDA API layer. 

And NVIDIA publishes all of that source code openly. Everything I needed was right there.

That code was invaluable for getting the DMA-BUF import path right.

Then on my first tries something was not working as expected, Ollama was still doing whatever it wanted, ignoring GreenBoost entirely, as if it wasn't there.

Exploring a bit further is when I found I should intercept Ollama call because that is what
I'm using nowadays, though I am aware the AI inference landscape is much broader. 

I explored documentation to get as much information on ollama project as I could reach, I was looking for how `llama.cpp` handles CPU offload, how ExLlamaV2/V3 structures its KV cache, and how vLLM's paged attention allocates memory. 

Gathering this information was decisive to know how those allocation lifecycles work and where in greenboost stack should be the code to intercept the call, I gathered different opensource libraries to get a broad scenario so that any inference tool would work without modification, not just Ollama, that had been the goal.

and that's it

Synapse Terminal (a non opensource local AI Inference terminal), started to answer my questions in a proper pace,
I was stonished at the beggining, could not believe glm-4.7-flash:q8_0 was working that relatively well (later improved). 

Another thing I had in mind is that the OS should be optimized as much as possible. That's why I created the tune-all command to optimize the system as much as possible for local AI inference. This command runs by default when choosing the full-install command.

I do have a modified (non official BIOS) that includes latest intel microcode, since the brand of my motherboard hasn't released any newer version, for you to have an idea on how much I like optimization.

There is still much more to do, I guess. I don't know when my hardware will surprise me again by pushing past what I thought were its limits.

I personally have much to learn about AI Inference. That's why this project has a tools section, to dive further into AI inference.

Hope you all enjoy greenboost kernel modules.

---

## Installation

```bash
git clone https://gitlab.com/IsolatedOctopi/nvidia_greenboost.git
cd nvidia_greenboost

# Auto-detects GPU VRAM, RAM size, CPU P/E-core topology, NVMe capacity
# and computes optimal parameters at install time:
sudo ./greenboost_setup.sh full-install

# You can skip the GitLab version check with:
sudo ./greenboost_setup.sh full-install --skip-update-check

# After reboot, verify everything is working:
sudo ./greenboost_setup.sh diagnose
```

The installer then runs the following steps automatically:

| Step | What it does |
|------|-------------|
| Version check | Queries GitLab tags API; if a newer release exists, offers to `git pull` and restart — skipped with `--skip-update-check` or when offline |
| Purge | Stops Ollama/llama-server, unloads any previous GreenBoost module, removes old install artifacts |
| Dependencies | Installs build tools, kernel headers, CUDA dev packages |
| Build + install | Compiles `greenboost.ko` + `libgreenboost_cuda.so`, installs to `/lib/modules` and `/usr/local/lib` |
| Load | Inserts the kernel module with auto-detected pool sizes (T1/T2/T3) |
| System configs | Writes Ollama systemd env vars, NVMe udev rules, CPU governor service, hugepages, sysctl |
| Tune-all | **CPU governor → performance; NVMe scheduler → none, read-ahead → 4 MB; THP → always; vm.swappiness=10; GRUB params (rcu_nocbs, nohz_full, transparent_hugepage=always); AI/compute libraries (OpenBLAS AVX2, hwloc, libnuma, nvtop, microcode)** |
| Tools | Creates `/opt/greenboost/venv` and installs Python inference tools: ExLlamaV3 (alternative inference engine with GreenBoost-aware KV cache), kvpress, LoRA fine-tuning utilities, and model optimization helpers |
| Restart | Restarts any services that were stopped during purge (Ollama, llama-server) |

`greenboost_setup.sh` detects Red Hat-based systems at runtime and delegates to
`greenboost_setup_rocky.sh` (contributed by Alan Sill, see contributors below).

---

## Quick usage

```bash
# Run Ollama as normal — GreenBoost is transparent
ollama run glm-4.7-flash:q8_0

# Monitor memory tiers live:
watch -n1 'cat /sys/class/greenboost/greenboost/pool_info'
```

For tools that require manual model loading, see
**[implementing_further_into_tools.md](implementing_further_into_tools.md)**.

Ollama does not need this; GreenBoost integrates with it automatically via the CUDA shim.

---

## What GreenBoost is not

- It is **not** a replacement for the NVIDIA driver. `nvidia.ko`, `nvidia-uvm.ko`, and all
  NVIDIA official modules continue to run exactly as normal. GreenBoost loads beside them.
- It is **not** a virtual GPU. It does not expose a new GPU device or change how compute
  works. It only affects how CUDA memory allocations are routed.
- It is **not** a hack around driver restrictions. The DMA-BUF + external memory import
  path it uses is a documented CUDA feature.
- It does **not** work without the NVIDIA driver installed.

---

## Usage reference

```
sudo ./greenboost_setup.sh [--skip-update-check] <command>

GLOBAL FLAGS:
  --skip-update-check  Skip GitLab version check (offline / air-gapped workstations)

COMMANDS:
  install              Build and install module + CUDA shim system-wide
  uninstall            Unload, remove module + all config files
  build                Build only (no system install)
  load                 Load module with default 3-tier parameters
  unload               Unload module (keeps installed files)
  tune                 Tune system for LLM workloads (governor, NVMe, THP, sysctl)
  tune-grub            Fix GRUB boot params (THP=always, rcu_nocbs, nohz_full…)
  tune-sysctl          Consolidate sysctl files + apply compute-optimized knobs
  tune-libs            Install missing AI/compute libraries (OpenBLAS, hwloc…)
  tune-all             Run tune + tune-grub + tune-sysctl + tune-libs in sequence
  install-sys-configs  Install Ollama env, NVMe udev, CPU governor, hugepages, sysctl
  install-deps         Install all Ubuntu OS packages (build + CUDA + AI libs)
  setup-swap [GB]      Create/activate NVMe swap (default: auto-sized)
  full-install         Complete install — deps, module, shim, configs, tune-all, Python inference tools
  status               Show module status and 3-tier pool info
  diagnose             Full health check — run this after reboot to verify everything works
  optimize-model [--model M] [--strategy tensorrt|lora|exllama|all]
                       Optimize LLM for max speed: TRT-LLM, LoRA, ExLlamaV3
  help                 Show this help

CUDA SHIM WRAPPER:
  greenboost run <app> [args...]   Run a CUDA app with GreenBoost overflow enabled
```
---

## Using GreenBoost inside containers, VMs, and WSL2

On bare metal, the greenboost kernel module  handles everything. For environments where `greenboost.ko` cannot be loaded; Docker, LXC, KVM guests, WSL2, shared HPC clusters, the CUDA shim automatically falls back to Path B: `cuMemHostRegister(DEVICEMAP)`, which
pins anonymous system RAM pages directly through the CUDA driver with no kernel module required. Check Jerry Nguyen's contribution (see Contributors), for full explanation of path selection, bandwidth trade-offs, and per-environment instructions see **[container_vm_mode.md](container_vm_mode.md)**.

---

## Model optimization tools

The [`tools/`](tools/) directory bundles several open-source libraries for further tuning; quantization, KV cache compression, fine-tuning, and alternative inference engines. See **[tools/README.md](tools/README.md)** for details.

---

## License

GPL v2 — open-source. Attribution to Ferran Duarri is required in all forks, derivatives,
and any documentation that references this work.

```
Copyright (C) 2026 Ferran Duarri
```

---

## Contributors

- **Alan Sill** ([@alansill](https://gitlab.com/alansill)) ; contributed `greenboost_setup_rocky.sh`,
  a setup script for Red Hat-based systems (Rocky Linux, AlmaLinux, RHEL).

- **Jerry Nguyen** ([@phubao](https://gitlab.com/phubao)) ; contributed the
  kernel-module-free overflow path, the `cuMemHostRegister(DEVICEMAP)` approach that
  enables GreenBoost VRAM extension inside containers, VMs, WSL2, and HPC clusters without
  requiring `greenboost.ko`. Integrated as Path B of the blended shim in v2.5.