# B10 MetaRoCE (including the MRC protocol) — spec digest + build record

## 1. Project information

| Item | Value |
|---|---|
| repository | `github.com/facebookincubator/metoce` → actually `facebookincubator/meta**roce**` |
| license | Apache-2.0 |
| language | C++ |
| size | 10 MB (source) / spec **6,095 lines** in `spec/MetaRoCE-Spec-1.0.md` + PDF |
| components | `spec/` (the specification), `libsoftmetaroce/` (software transport + verbs provider), `rdma_gen/` (RDMA traffic generator + Thrift control plane), `build/` (Meta getdeps build system) |
| positioning | an RDMA transport aimed at **million-GPU clusters** running on **lossy Ethernet**, replacing standard RoCEv2 |
| ecosystem relation | complementary to B9's Open-RDMA: Open-RDMA provides a full open-source RoCEv2 stack, MetaRoCE provides a **multipath, loss-tolerant** transport layer |

## 2. Four core mechanisms (spec §1.3)

| Mechanism | Content | Why it matters |
|---|---|---|
| **Tag connection multiplexing** | one QP carries multiple independent message streams (Tags), each with its own sequence-number space and completion order | one stream stalled by loss / a not-ready receiver **does not block** the others |
| **Explicit multipath** | the sender maintains multiple flows inside one connection, each with an independent sliding window + SACK state; different ECMP routes are reached by changing the packet-header entropy | flow state uses **fixed-length bitmaps**, so memory is bounded; **it distinguishes "this flow is dead" from "temporary congestion"** (blind spraying cannot); when the bandwidth-delay product grows, **add flows rather than enlarging a single flow's window** |
| **Relaxed in-order delivery** | out-of-order packets are written straight to their final memory location; completion notifications are given to the application in order | each packet carries its in-message offset; Writes also carry address/length/rkey; **no reorder buffer on the NIC that must grow with the BDP** |
| **Receiver-driven rate-limited congestion control** | baseline AIMD responding to ECN and loss + the receiver returning "your inbound bandwidth share" in the ACK | incast converges in **1–2 RTTs**, with no dependence on PFC |

## 3. Supported operations (spec §1.5)

1. SEND / RECV (with/without immediate, with/without Tag)
2. RDMA Write (with/without immediate)
3. RDMA Read (with/without immediate)
4. Atomics: compare-and-swap, fetch-and-add (opcodes reserved for bitwise operations etc.)

## 4. Packet header format (spec §8, directly usable for a from-scratch implementation)

**Base Header** (4-byte aligned, MSB order):

```
 0         1         2         3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+---------------+---------------+-------+-----------------------+
|    Opcode     |   Reserved    |Version|        FlowID         |
+---------------------------------------------------------------+
|                        Destination QP                         |
+---------------------------------------------------------------+
|                           Source QP                           |
+---------------------------------------------------------------+
```

- official example code for extracting FlowID/Opcode:
  ```c
  uint32_t w = ntohl(((uint32_t *)p)[0]);
  int path_id = w & 0xfff;      // low 12 bits = FlowID
  int opcode  = w >> 24;        // high 8 bits = Opcode
  ```
- Opcode: bits 7:5 of the RoCEv2 opcode use `110` to indicate Meta-RoCE Connected Mode (the vendor-defined
  range), bits 4:0 follow the RoCEv2 scheme; of RoCEv2's first/middle/last three states **only first is used**,
  the rest are reserved.
- Meta-RoCE **defines its own** METH (data), SAETH (selective ACK) and RNR-Cancel headers;
  it **borrows from RoCEv2** RETH (read/write), ImmDt, IETH, AtomicETH, AtomicAckETH.
- every packet (data/ACK/control) carries a trailing **32-bit mCRC** covering header + payload + padding;
  mCRC is **byte-granular**, so the receiver allows payloads not aligned to 4 bytes.
- the sender must pad the payload to a multiple of 4 bytes (and, when running directly over Ethernet, satisfy the
  minimum frame length); the padding byte count is recorded in the data header's Padding field.
- reserved fields: the sender must write 0, and **the receiver must not reject a packet because of them**.

## 5. Published performance data (Meta's figures, 64-node AMD GPU cluster, RCCL collective communication)

| Metric | RoCEv2 | MetaRoCE |
|---|---|---|
| 0% loss throughput | baseline | higher |
| **1% loss** | marked degradation | **≈86% throughput** |
| **10% loss** | collapse | **still usable** |
| multi-plane scaling | limited | **linear (4/8 planes verified, up to 4,000 concurrent connections)** |
| single-plane failure recovery | manual / ops | **autonomous reallocation, transparent to the application** |

It only requires two capabilities that deployed switches already have: **ECN marking + ECMP**;
it does not require packet trimming, in-band telemetry, credit-based flow control, or switch-side spraying →
**topology-independent** (fat-tree / multi-plane / deep-buffered / shallow-buffered / uncontrolled vendor clouds
all work).

## 6. Build attempt (getdeps)

Upstream's only build entry point:

```bash
python3 build/fbcode_builder/getdeps.py build metaroce --src-dir .
# i.e. ./build.sh in the repository root
```

**Hard constraint**: the default compiler must be **gcc, not clang**, otherwise **folly/coro will not be built**.
Dependency chain: `folly` (with coro), `glog`, `gtest`, `thrift` (rdma_gen control plane) + gflags etc.

Conditions on this machine:

| Item | Value |
|---|---|
| gcc | 12.2.0 (Debian 12) ✓ satisfies "must be gcc" |
| cmake | 3.25.1 |
| disk | /root has ample space (source is only 10 MB, but getdeps builds folly etc., needing several GB) |
| network | the NAS has internet access (see the correction in B9) ✓ |

Build log: `/root/b10-build.log` (NAS)

### 6.1 First build: stuck on a libaio download 404 (root cause located and fixed)

Failed after 23m44s; gflags / glog both **compiled successfully**, then it hung on:

```
Failed to download https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
HTTP Error 404: Not Found
```

**Root cause found by searching online** (not a network problem):
> **pagure.io code hosting was shut down in mid-2026**, so libaio's archive download path became invalid.
> libaio upstream moved to Codeberg: `jmoyer/libaio` (maintained by Jeff Moyer),
> release asset `https://codeberg.org/jmoyer/libaio/releases/download/libaio-0.3.113/libaio-0.3.113.tar.gz`.
> Note that the Codeberg tarball and the old pagure archive **are not the same file** (md5 `7d5be185…` vs the old
> `605237f3…`), because pagure packed with git-archive and the directory layout differs.

And this repository's `build/fbcode_builder/manifests/libaio` pins the old package's hash:
```toml
[manifest]
name = libaio
[debs]
libaio-dev          # ← key: the manifest itself declares a usable system package
[download]
url = https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
sha256 = 2c44d1c5fd0d43752287c9ae1eb9c023f04ef848ea8d4aafa46e9aedb678200b
```

**Fix (using a mechanism upstream itself supports, without changing any pin value)**:
```bash
apt install -y libaio-dev
python3 build/fbcode_builder/getdeps.py --allow-system-packages build metaroce --src-dir .
```
i.e. let getdeps take the `[debs]` branch and use the system `libaio-dev`, skipping the broken download entirely.

> Lesson (consistent with the 2026-10-20 01:46 entry): **the first step on any error is to search the error
> message.** Without searching, one would blame "a NAS network problem" or hand-edit the hash — while the real
> cause was **the upstream code-hosting site being shut down**.

### 6.2 The build chain has four traps in total (all located, with fixes)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `libaio-0.3.113.tar.gz` **404** | **pagure.io shut down in mid-2026**, archive path invalid | `apt install libaio-dev` + `--allow-system-packages` (the manifest ships `[debs] libaio-dev`) |
| 2 | `automake-1.16.1.tar.gz` **403** | `ftpmirror.gnu.org` returns 403 for getdeps' UA/IP | fetch the same file from `cdimage.debian.org/mirror/gnu.org/gnu/...`, **verify against the manifest's sha256**, then place it in getdeps' `downloads/` directory (filename format `<dep>-<tarball name>`) |
| 3 | similar 403s keep recurring | this is **systemic**, not a single package | one-shot script: parse every manifest `url` + `sha256` pointing at `ftpmirror.gnu.org`/`ftp.gnu.org` and **fill them all in a batch** (5 this time: binutils-2.43 / gperf-3.3 / libtool-2.4.6 / ncurses-6.3, all hashes matching) |
| 4 | folly link failure: `relocation R_X86_64_PC32 ... recompile with -fPIC`, `relocation against nil_eventop` | the `libevent` manifest has **`EVENT__LIBRARY_TYPE = STATIC` with no PIC definition** → `libevent.a` is non-PIC and cannot be linked into `libfolly.so` | ① use getdeps' built-in **`--shared-lib`** ("Build deps as static-**PIC** archives"); ② if still insufficient, add **`CMAKE_POSITION_INDEPENDENT_CODE = ON`** explicitly to `manifests/libevent`'s `[cmake.defines]`; ③ **you must clear the `build/` and `installed/` caches** (getdeps reuses old artifacts by hash directory; without clearing, the old non-PIC `.a` is used forever; keep `downloads/` to avoid re-downloading) |

**Pitfalls to write down**:
- manifest filenames have **no extension** (`manifests/libevent`, not `libevent.toml`), so `grep "*.toml"` misses them all.
- when clearing caches **keep `downloads/`**: it holds the hash-verified original tarballs; deleting them means
  re-downloading (binutils alone is 28 MB).
- `cp` a backup before editing any manifest (this time to `/root/libevent.manifest.bak`).
- to tell "stale cache" from "a real error": check whether the `installed/<dep>-<hash>/` hash directory in the
  error **changed**. Unchanged hash = old artifacts being reused.

## 7. Relation to the MX450 project (honest assessment)

| Dimension | Conclusion |
|---|---|
| can it run on this machine | depends on whether getdeps can build folly/coro under Debian 12 + gcc12 |
| **does it help GPU compute** | **No.** It is a network transport layer; it does not change the MX450's SM / VRAM / clocks |
| significance for F3 "rewrite the network stack" | **directly relevant** — F3 has always been ❌ untouched, and MetaRoCE's spec (sequence-number hierarchy, SACK bitmaps, out-of-order delivery, receiver-side rate limiting) **is exactly the implementation blueprint for "the network stack"** |
| combination with B9 Open-RDMA | Open-RDMA gives "an open verbs stack + mock device", MetaRoCE gives "multipath loss-tolerant transport semantics" — only stacked together do they form a complete, modifiable software-RDMA path |

**→ So the correct delivery for B10 is not the sentence "it compiles", but:**
1. really run the upstream build (in progress);
2. **understand the spec and implement a minimally runnable MRC semantic in our own code** (Tag multi-stream +
   per-flow independent windows + 256-bit SACK bitmap + out-of-order commit + receiver rate hints),
   directly as the first brick of **F3 "rewrite the network stack"**.

## 8. Related search points (web-verified)

- Meta published the design details on 2026-08-24, and at the 2026-10 OCP summit released the **complete spec +
  `libsoftmetaroce` reference implementation + production compliance test suite**.
- MRC = **Multipath Reliable Connection**, jointly proposed by OpenAI / AMD / Broadcom / Microsoft / NVIDIA;
  the Open-RDMA project also states that its multipath transport borrows MRC mechanisms (see B9's `open-rdma`
  main repo README).
- So B10 really requires reading **the two projects side by side**: MRC is the protocol semantics, MetaRoCE is
  Meta's implementation, Open-RDMA is another open-source implementation.

## 9. fabric-lib (perplexityai/pplx-garden) — real-run record 2026-10-20

### 9.1 Positioning

fabric-lib is **not a standalone repository**; it is open-sourced inside `github.com/perplexityai/pplx-garden`
(MLSys 2026 oral, arXiv:2510.27656, MIT, C++/Rust).
It unifies the P2P interface of ConnectX and EFA RDMA: WRITEIMM + IMM COUNTER completion notification (no ordering
assumption), multi-NIC aggregation, transparent multipath. Published data: 400 Gbps on ConnectX-7 and EFA.

### 9.2 Build chain (all really run and passed on this NAS)

| Step | Result |
|---|---|
| `cargo build --release --bin fabric-debug` | stuck on missing `cuda_profiler_api.h` (this machine's slim CUDA 12.9 install lacks it) |
| fix 1: wrap that include in `cudart-sys/wrapper.h` with `#if __has_include` | passed (backup at `/root/wrapper.h.bak.cudart-sys`) |
| fix 2: `GDRAPI_HOME` missing → installed **GDRCopy v2.5** from source (`make prefix=/usr/local CUDA=/usr/local/cuda install`) | passed |
| fix 3: `FI_MR_DMABUF`/`FI_OPT_CUDA_API_PERMITTED` unresolved → these are **AWS EFA extension constants**, absent from Debian's standard libfabric → installed the **AWS efa-installer** (`--skip-kmod --skip-limit-conf --no-verify`, user-space only, libfabric lands in `/opt/amazon/efa`) | **build succeeded: `target/release/fabric-debug`, a 2,970,264 B ELF** |

### 9.3 Smoke run (honest record)

```
$ fi_info -t FI_EP_RDM     # tcp provider sees enp45s0/wlp47s0/blue3; bluerdma0-3 still present
$ ./fabric-debug 0 1
Error: FabricLibError: No visible NICs
```

**Conclusion (the same kind of "real reality" as B8 Sudoku)**:
- interface / dependencies / build chain **all really work** (this is not a "judged unusable" case);
- at runtime fabric-lib's NIC enumeration recognises EFA/standard verbs devices and **does not recognise B9's mock
  `bluerdma`**, and this machine has no ConnectX/EFA → the data path cannot be reached here.
- valuable actionable margin: ① make fabric-debug use B9's bluerdma device (requires adding provider recognition
  to fabric-lib's NIC enumeration — that falls into "read the source and reimplement with our own code");
  ② the purely software part (WRITEIMM / IMM COUNTER semantics) can be implemented and verified in our own code.

### 9.4 Relation to the MX450 (honest assessment)

fabric-lib targets **400G datacentre RDMA**, and the MX450 host's PCIe 3.0 x4 (~4 GB/s) is outside its target
surface. Its value in this project is: **the second brick of the F3 network stack + an interface-design reference
aligned with MRC/Open-RDMA**, not a compute-improvement path.
