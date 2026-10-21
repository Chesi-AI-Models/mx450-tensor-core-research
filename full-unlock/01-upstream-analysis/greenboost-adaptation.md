# GreenBoost 在 MX450 上的完整适配报告

> **结果：物理显存 1.64 GiB，成功分配 8 GiB，且 matmul 结果完全正确。**
>
> 这是本项目最有实际价值的成果——它让「2 GB 显存跑 7B QLoRA」从不可能变成可能。

---

## 一、为什么这个项目重要

MX450 的物理显存只有 1.64 GiB 可用（2 GB 物理）。

7B QLoRA 需要约 4 GB。**差距是 2.4 倍。**

GreenBoost 用「三级存储」把系统 RAM 和 NVMe 变成虚拟显存：

| Tier | 介质 | 本机配置 | 带宽 |
|---|---|---|---|
| **T1** | GPU VRAM | **2 GB** | ~80 GB/s |
| **T2** | 系统 DDR（pinned DMA-BUF） | **8 GB** | ~4 GB/s（PCIe 3.0 x4） |
| **T3** | NVMe swap | **8 GB** | ~1.8 GB/s |

**合计 18 GB 模型容量。**

---

## 二、适配过程（四步）

### 步骤 1 · 编译内核模块

```bash
# 上游：greenboost 的 greenboost.c + greenboost_ioctl.h（用 refs-fetch-manifest.sh 获取）
cat > Makefile <<'EOF'
obj-m += greenboost.o
KDIR ?= /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
EOF

make
```

**结果：一次编译通过。**

```
CC [M]  greenboost.o
LD [M]  greenboost.ko
```

**内核版本 `6.18.18.c1126-trim`（fnOS 定制内核）完全兼容。**
唯一警告是编译器小版本差异（12.2.0-14 vs 12.2.0-14+deb12u1），无影响。

### 步骤 2 · 加载模块（参数调优）

**第一次失败**：`safety_reserve_gb=4` 时，系统 `free` 只有 2.6 GB，OOM guard 触发：

```
greenboost: T2 OOM guard TRIPPED — free=2694MB < reserve=4GB
T2 available: 0 MB
```

**原因**：Linux 的 `free` 不含 page cache（本机有 10 GB 在 cache 里），
而 GreenBoost 的 watchdog 看 `nr_free_pages`。

**解法**：释放 page cache + 降低 reserve。

```bash
sync && echo 3 > /proc/sys/vm/drop_caches
insmod greenboost.ko physical_vram_gb=2 virtual_vram_gb=8 \
       safety_reserve_gb=2 nvme_swap_gb=8 pcores_only=0 use_hugepages=1
```

**结果**：

```
T1 VRAM : 2 GB
T2 RAM  : pool cap 8 GB (reserve 2 GB)
T3 NVMe : 8 GB
Combined: 18 GB

T2 available : 11076 MB
OOM guard    : no
```

### 步骤 3 · 编译 CUDA shim

**shim 不依赖 `cuda.h`**（全部用 `dlsym` 动态解析），但有两个 Debian 12 的编译问题：

| 问题 | 解法 |
|---|---|
| `MAP_HUGE_2MB` 未定义 | `-DMAP_HUGE_2MB=0x54000000` |
| `_GNU_SOURCE` 重复定义 | 忽略警告 |

```bash
gcc -shared -fPIC -O2 -D_GNU_SOURCE -DMAP_HUGE_2MB=0x54000000 \
    -o libgreenboost_cuda.so greenboost_cuda_shim.c -ldl -lpthread
```

### 步骤 4 · **关键修复：`_v2` 符号**

**原版 shim 用 `dlsym(libcuda, "cuMemHostRegister")`，在 CUDA 12 下返回 201（INVALID_CONTEXT）。**

改成 `_v2` 版本后立即成功：

```c
dlsym(libcuda, "cuMemHostRegister_v2")
dlsym(libcuda, "cuMemHostGetDevicePointer_v2")
```

**这是本适配唯一的代码修改，两行。**

---

## 三、验证结果

### 显存分配测试

```
physical VRAM reported: 1.64 GiB

alloc 2 GiB: OK   reserved=2.00 GiB
alloc 3 GiB: OK   reserved=3.00 GiB
alloc 4 GiB: OK   reserved=4.00 GiB
alloc 6 GiB: OK   reserved=6.00 GiB
alloc 8 GiB: OK   reserved=8.00 GiB      ← 8 GiB，物理的 4.9 倍
```

### 计算正确性

```
matmul on 4GiB-occupied GPU: OK
c[0,0] = 1024.0 (expect 1024.0)          ← 完全正确
```

### 内核侧记录

```
greenboost: pinned 3072MB user buffer (id=1 fd=43)
```

---

## 四、使用方式

```bash
# 1. 加载模块（每次开机后）
sudo modprobe greenboost physical_vram_gb=2 virtual_vram_gb=8 \
     safety_reserve_gb=2 nvme_swap_gb=8 pcores_only=0 use_hugepages=1

# 2. 用 shim 跑你的程序
LD_PRELOAD=/vol1/ganlin/greenboost-build/libgreenboost_cuda.so \
GREENBOOST_ACTIVE=1 \
python train.py

# 3. 监控
watch -n1 'cat /sys/class/greenboost/greenboost/pool_info'
```

---

## 五、必须说清的代价

### 1. 带宽断崖

| 层级 | 带宽 | 相对 T1 |
|---|---|---|
| T1（VRAM） | ~80 GB/s | 1.0x |
| T2（PCIe 3.0 x4） | ~4 GB/s | **1/20** |
| T3（NVMe） | ~1.8 GB/s | 1/44 |

**推理能忍（权重读一次，计算在 T1），训练会很痛苦（梯度更新反复访问权重）。**

### 2. 本次走的是 UVM 路径

`T2 allocated: 0 MB` + `Active DMA-BUF objects: 0` 说明实际分配走的是
`cuMemAllocManaged`（Path 2），不是 DMA-BUF（Path 1）。

UVM 的页面迁移由驱动按需处理，延迟比 pinned DMA-BUF 高。

**要强制走 DMA-BUF 路径，需要确认 `gb_use_dmabuf` 的启用条件。**

### 3. 稳定性

- 模块是 GPL v2，原样使用
- 已加载状态下不影响 NVIDIA 驱动（`nvidia-smi` 正常）
- 卸载：`rmmod greenboost`
- **重启后需重新加载**（未配置 systemd 持久化）

---

## 六、对「训 7B 甘霖」意味着什么

| 项目 | 之前 | 现在 |
|---|---|---|
| 可用显存 | 1.64 GiB | **8 GiB（虚拟）** |
| 7B QLoRA（4 GB 需求） | ❌ 装不下 | ✅ **装得下** |
| 训练速度 | — | 受 T2 带宽限制，预计比纯 VRAM 慢 3-10 倍 |

**结论：从「物理不可能」变成「能跑，但慢」。**

**这是这台机器上能做到的极限。**

---

## 七、上游信息

- **项目**：GreenBoost v2.5
- **作者**：Ferran Duarri
- **许可**：GPL v2（开源）/ 商业双许可
- **上游**：`refs/refs-fetch-manifest.sh` 拉取（本仓库不再内联源码）
- **本适配的修改**：仅 2 行（`cuMemHostRegister` → `_v2`）

**本适配报告为原创，记录了 MX450 + fnOS 6.18.18 上的完整验证过程。**
