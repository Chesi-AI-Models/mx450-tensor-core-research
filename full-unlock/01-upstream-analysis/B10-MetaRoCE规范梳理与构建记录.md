# B10 MetaRoCE（含 MRC 协议）—— 规范梳理 + 构建记录

## 一、项目信息

| 项 | 值 |
|---|---|
| 仓库 | `github.com/facebookincubator/metoce` → 实际为 `facebookincubator/meta**roce**` |
| 许可 | Apache-2.0 |
| 语言 | C++ |
| 规模 | 10 MB（源码）/ 规范 **6095 行** `spec/MetaRoCE-Spec-1.0.md` + PDF |
| 组件 | `spec/`（规范）、`libsoftmetaroce/`（软件传输 + verbs provider）、`rdma_gen/`（RDMA 流量生成器 + Thrift 控制面）、`build/`（Meta getdeps 构建系统） |
| 定位 | 面向**百万 GPU 集群**、运行在**会丢包的以太网**上的 RDMA 传输协议，取代标准 RoCEv2 |
| 生态关系 | 与本项目 B9 的 Open-RDMA 互补：Open-RDMA 提供开源 RoCEv2 全栈，MetaRoCE 提供**多路径 + 抗丢包**的新传输层 |

## 二、四个核心机制（规范 §1.3）

| 机制 | 内容 | 为什么重要 |
|---|---|---|
| **Tag 连接复用** | 一个 QP 承载多条独立消息流（Tag），各有自己的序列号空间与完成顺序 | 一条流因丢包/接收端未就绪卡住，**不阻塞**其他流 |
| **显式多路径** | 发送端在一个连接内维护多条 flow，每条有独立滑动窗口 + SACK 状态；靠改包头 entropy 走不同 ECMP 路由 | flow 状态用**定长位图**，内存有界；**能区分"这条 flow 挂了"和"暂时拥塞"**（盲目喷洒做不到）；带宽时延积变大时**加 flow 而不是加大单流窗口** |
| **松弛顺序交付** | 数据包乱序到达即直接写入内存最终位置；完成通知按序给应用 | 每个包自带消息内偏移；Write 还带地址/长度/rkey；**不需要网卡上随 BDP 增长的重排序缓冲** |
| **接收端驱动限速的拥塞控制** | 基线 AIMD 响应 ECN 与丢包 + 接收端在 ACK 里回传"分给你的入站带宽份额" | incast 场景**1–2 个 RTT 收敛**，彻底不依赖 PFC |

## 三、支持的操作（规范 §1.5）

1. SEND / RECV（带/不带 immediate、带/不带 Tag）
2. RDMA Write（带/不带 immediate）
3. RDMA Read（带/不带 immediate）
4. Atomics：compare-and-swap、fetch-and-add（opcode 为位运算等预留）

## 四、包头格式（规范 §8，可直接用于从零实现）

**Base Header**（4 字节对齐，MSB 序）：

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

- 取 FlowID/Opcode 的官方示例代码：
  ```c
  uint32_t w = ntohl(((uint32_t *)p)[0]);
  int path_id = w & 0xfff;      // 低 12 位 = FlowID
  int opcode  = w >> 24;        // 高 8 位 = Opcode
  ```
- Opcode：RoCEv2 opcode 的 bits 7:5 用 `110` 表示 Meta-RoCE Connected Mode（厂商自定义区），
  bits 4:0 沿用 RoCEv2 模式；RoCEv2 的 first/middle/last 三态中**只采用 first**，其余保留。
- Meta-RoCE **自定义** METH（数据）、SAETH（选择性 ACK）、RNR-Cancel 头；
  **借用 RoCEv2** 的 RETH（读写）、ImmDt、IETH、AtomicETH、AtomicAckETH。
- 所有包（数据/ACK/控制）尾部都有 **32 位 mCRC**，覆盖头 + 载荷 + padding；
  mCRC 是**字节粒度**，因此接收端允许载荷不按 4 字节对齐。
- 发送端需把载荷 pad 到 4 字节倍数（直跑以太网时还要满足最小帧长），pad 字节数记在数据头的 Padding 字段。
- 保留字段：发送方必须写 0，**接收方不得因此拒包**。

## 五、公开性能数据（Meta 口径，64 节点 AMD GPU 集群，RCCL 集合通信）

| 指标 | RoCEv2 | MetaRoCE |
|---|---|---|
| 0% 丢包吞吐 | 基线 | 更高 |
| **1% 丢包** | 明显退化 | **≈86% 吞吐** |
| **10% 丢包** | 崩 | **仍可用** |
| 多平面扩展 | 有限 | **线性（4/8 平面已验证，最多 4000 条并发连接）** |
| 单平面失效恢复 | 人工/运维 | **自主重分配，应用无感** |

它只要求网络提供两件现网交换机本来就有的能力：**ECN 标记 + ECMP**；
不要求包裁剪、带内遥测、信用流控、交换机侧喷洒 → **拓扑无关**（胖树/多平面/深缓冲/浅缓冲/不可控的厂商云均可）。

## 六、构建尝试（getdeps）

上游唯一构建入口：

```bash
python3 build/fbcode_builder/getdeps.py build metaroce --src-dir .
# 即仓库根目录的 ./build.sh
```

**硬约束**：默认编译器必须是 **gcc 而非 clang**，否则 **folly/coro 不会被构建**。
依赖链：`folly`（含 coro）、`glog`、`gtest`、`thrift`（rdma_gen 控制面）+ gflags 等。

本机条件：

| 项 | 值 |
|---|---|
| gcc | 12.2.0（Debian 12）✓ 满足"必须是 gcc" |
| cmake | 3.25.1 |
| 磁盘 | /root 空间充足（源码仅 10MB，但 getdeps 会构建 folly 等，需数 GB） |
| 网络 | NAS 有外网（见 B9 的纠正）✓ |

构建日志：`/root/b10-build.log`（NAS）

### 6.1 第一次构建：卡在 libaio 下载 404（已定位根因并修复）

23m44s 后失败，gflags / glog 均**已成功编译**，卡在：

```
Failed to download https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
HTTP Error 404: Not Found
```

**联网查到的根因**（不是网络问题）：
> **pagure.io 代码托管在 2026 年中关停**，libaio 的归档下载路径随之失效。
> libaio 上游迁到 Codeberg：`jmoyer/libaio`（Jeff Moyer 维护），
> 发布资产 `https://codeberg.org/jmoyer/libaio/releases/download/libaio-0.3.113/libaio-0.3.113.tar.gz`。
> 注意 Codeberg 的 tarball 与旧 pagure 归档**不是同一个文件**（md5 `7d5be185…` vs 旧 `605237f3…`），
> 因为 pagure 是 git-archive 打包，目录布局不同。

而本仓库 `build/fbcode_builder/manifests/libaio` 里钉死的是旧包的哈希：
```toml
[manifest]
name = libaio
[debs]
libaio-dev          # ← 关键：清单自己声明了可用的系统包
[download]
url = https://releases.pagure.org/libaio/libaio-0.3.113.tar.gz
sha256 = 2c44d1c5fd0d43752287c9ae1eb9c023f04ef848ea8d4aafa46e9aedb678200b
```

**修复方案（用上游自己支持的机制，不改任何 pin 值）**：
```bash
apt install -y libaio-dev
python3 build/fbcode_builder/getdeps.py --allow-system-packages build metaroce --src-dir .
```
即让 getdeps 走 `[debs]` 分支用系统 `libaio-dev`，完全跳过这个坏掉的下载。

> 教训（与 2026-10-20 01:46 那条一致）：**报错第一步是搜索错误信息**。
> 若不搜，会误以为是"NAS 网络问题"或去手动改哈希 —— 而真正原因是**上游代码托管站关停**。

### 6.2 构建链路一共四个坑（全部已定位，含修法）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `libaio-0.3.113.tar.gz` **404** | **pagure.io 2026 年中关停**，归档路径失效 | `apt install libaio-dev` + `--allow-system-packages`（清单自带 `[debs] libaio-dev`） |
| 2 | `automake-1.16.1.tar.gz` **403** | `ftpmirror.gnu.org` 对 getdeps 的 UA/IP 返回 403 | 从 `cdimage.debian.org/mirror/gnu.org/gnu/...` 取同文件，**按清单 sha256 校验**后放进 getdeps `downloads/` 目录（文件名格式 `<dep>-<tarball名>`） |
| 3 | 同类 403 会反复出现 | 这是**系统性**问题，不是单包 | 一次性脚本：解析全部清单里指向 `ftpmirror.gnu.org`/`ftp.gnu.org` 的 `url` + `sha256`，**批量补齐**（本次 5 个：binutils-2.43 / gperf-3.3 / libtool-2.4.6 / ncurses-6.3，全部哈希匹配） |
| 4 | folly 链接失败：`relocation R_X86_64_PC32 ... recompile with -fPIC`、`relocation against nil_eventop` | `libevent` 清单里 **`EVENT__LIBRARY_TYPE = STATIC` 且无 PIC 定义** → `libevent.a` 非 PIC，不能链进 `libfolly.so` | ① 用 getdeps 自带的 **`--shared-lib`**（"Build deps as static-**PIC** archives"）；② 仍不够时，给 `manifests/libevent` 的 `[cmake.defines]` 显式加 **`CMAKE_POSITION_INDEPENDENT_CODE = ON`**；③ **必须清掉 `build/` 与 `installed/` 缓存**（getdeps 按 hash 目录复用旧产物，不清会一直用非 PIC 的旧 `.a`；保留 `downloads/` 免得重下） |

**踩坑要点（写死）**：
- 清单文件名是 **无扩展名** 的（`manifests/libevent` 不是 `libevent.toml`），`grep "*.toml"` 会全部漏掉。
- 清缓存时**保留 `downloads/`**：里面是校验过哈希的原始 tarball，删了要重下一遍（binutils 就 28MB）。
- 每次改清单前先 `cp` 备份（本次备份到 `/root/libevent.manifest.bak`）。
- 判断"是缓存脏了还是真错"：看报错里的 `installed/<dep>-<hash>/` 哈希目录**有没有变**。
  哈希不变 = 复用了旧产物。

## 七、与 MX450 项目的关系（诚实评估）

| 维度 | 结论 |
|---|---|
| 能否在本机跑通 | 取决于 getdeps 能否在 Debian 12 + gcc12 下把 folly/coro 构建出来 |
| **对 GPU 算力有没有帮助** | **没有**。它是网络传输层，不改变 MX450 的 SM/显存/时钟 |
| 对 F3「重写网络底层」的意义 | **直接相关** —— F3 一直是 ❌ 未碰，MetaRoCE 的规范（序列号层级、SACK 位图、乱序交付、接收端限速）**正是"网络底层"的实现蓝图** |
| 与 B9 Open-RDMA 的组合 | Open-RDMA 给「开源 verbs 栈 + mock 设备」，MetaRoCE 给「多路径抗丢包传输语义」——两者叠起来才是一条完整可改的软 RDMA 路径 |

**→ 因此 B10 的正确交付不是"编译通过"一句话，而是：**
1. 真跑上游构建（进行中）；
2. **读懂规范并用自研代码实现最小可跑的 MRC 语义**（Tag 多流 + 每 flow 独立窗口 + 256 位 SACK 位图 + 乱序落盘 + 接收端速率提示），
   直接作为 **F3「重写网络底层」** 的第一块砖。

## 八、相关搜索要点（联网核对）

- Meta 于 2026-08-24 公开设计细节，2026-10 OCP 峰会放出**完整规范 + `libsoftmetaroce` 参考实现 + 生产合规测试集**。
- MRC = **Multipath Reliable Connection**，由 OpenAI / AMD / Broadcom / Microsoft / NVIDIA 共同提出；
  Open-RDMA 项目也声明其多路径传输借鉴 MRC 机制（见 B9 的 `open-rdma` 主仓 README）。
- 所以 B10 实际上要把**两个项目对齐看**：MRC 是协议语义，MetaRoCE 是 Meta 的实现，Open-RDMA 是另一套开源实现。

## 九、fabric-lib（perplexityai/pplx-garden）—— 2026-10-20 真跑记录

### 9.1 定位

fabric-lib **不是独立仓库**，开源于 `github.com/perplexityai/pplx-garden`
（MLSys 2026 oral，arXiv:2510.27656，MIT，C++/Rust）。
统一 ConnectX/EFA 两家 RDMA 的 P2P 接口：WRITEIMM + IMM COUNTER 完成通知（无顺序假设）、
多 NIC 聚合、透明多路径。公开数据：ConnectX-7 与 EFA 上 400 Gbps。

### 9.2 构建链（本机 NAS 全部真跑通过）

| 步骤 | 结果 |
|---|---|
| `cargo build --release --bin fabric-debug` | 卡在 `cuda_profiler_api.h` 缺失（本机 CUDA 12.9 精简安装无此头） |
| 修 1：`cudart-sys/wrapper.h` 用 `#if __has_include` 包住该头 | 越过（备份 `/root/wrapper.h.bak.cudart-sys`） |
| 修 2：`GDRAPI_HOME` 缺失 → 从源码装 **GDRCopy v2.5**（`make prefix=/usr/local CUDA=/usr/local/cuda install`） | 越过 |
| 修 3：`FI_MR_DMABUF`/`FI_OPT_CUDA_API_PERMITTED` 未解析 → 这些是 **AWS EFA 扩展常量**，Debian 标准 libfabric 没有 → 装 **AWS efa-installer**（`--skip-kmod --skip-limit-conf --no-verify`，只装用户态，libfabric 落到 `/opt/amazon/efa`） | **编译成功：`target/release/fabric-debug` 2,970,264 B ELF** |

### 9.3 冒烟运行（诚实记录）

```
$ fi_info -t FI_EP_RDM     # tcp provider 可见 enp45s0/wlp47s0/blue3；bluerdma0-3 仍在
$ ./fabric-debug 0 1
Error: FabricLibError: No visible NICs
```

**结论（与 B8 Sudoku 同类的"真实现实"）**：
- 接口/依赖/编译链路**全部真跑通**（不是"判断用不了"）；
- 运行时 fabric-lib 的 NIC 枚举认 EFA/标准 verbs 设备，**不认 B9 的 mock `bluerdma`**，
  本机又没有 ConnectX/EFA → 数据通路在本机走不到。
- 有价值的可执行余量：① 让 fabric-debug 走 B9 的 bluerdma 设备（需在 fabric-lib 的
  NIC 枚举里加 provider 识别 —— 属于"读源码用自己的代码重实现"的范畴）；② 纯软件
  部分（WRITEIMM/IMM COUNTER 语义）可用自研代码实现并验证。

### 9.4 与 MX450 的关系（诚实评估）

fabric-lib 面向 **400G 数据中心 RDMA**，MX450 主机的 PCIe 3.0 x4（~4 GB/s）不在其目标面。
它的价值在本项目里是：**F3 网络底层的第二块砖 + 与 MRC/Open-RDMA 对齐的接口设计参照**，
不是算力提升路径。
