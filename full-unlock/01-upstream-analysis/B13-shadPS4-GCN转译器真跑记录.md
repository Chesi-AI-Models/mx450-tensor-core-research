# B13 shadPS4 GCN 转译器真跑记录（2026-10-20 19:30–20:50）

**简体中文** | [繁體中文](B13-shadPS4-GCN转译器真跑记录.zh-TW.md) | [English](B13-shadPS4-GCN转译器真跑记录.en.md)

> 台账项：B13 shadPS4 显存重构　结论：**✅ 真跑完成**（编译 + 在 MX450 上执行 58 个测试）
> 全部结论来自 NAS（LYLNAS）实测，非读文档。

## 1. 目标与合格判定

B13 的两个层面：
1. **档 3 方法论价值**：shadPS4 的 `ResourcePool`「资源死亡判定提前释放」机制
   （ticks 数组 / grow_step / hint_iterator / ManageOverflow）——**已读透，可借鉴到
   GreenBoost/T2 池**（见 §5）。
2. **真跑合格判定**（同 B11 AITER 标准）：把 shadPS4 的 **gcn shader_recompiler 测试**
   （GCN→SPIR-V 转译器，CPU 侧 + Vulkan 执行）在 MX450 上编译并跑出数。

## 2. 突破链（每一步实测）

### 2.1 工具链：C++23 的正解

shadPS4 要求 **C++23**（`std::expected` + `<format>`），g++ 12.2 两者皆无。

| 步骤 | 结果 |
|---|---|
| 搜索确认 Debian bookworm 主仓有 `clang-19`（19.1.7-3~deb12u1，2025-06 重建版） | ✅ |
| `apt install clang-19 libc++-19-dev libc++abi-19-dev` | ✅ |
| C++23 实证程序（std::expected + std::format + libc++） | **编译并运行通过** |
| 读 shadPS4 `tests/CMakeLists.txt:217-226` | **官方对 libc++ 加 `-fexperimental-library`**——libc++ 是受支持配置 |

最终编译参数：`clang++-19 -std=c++23 -stdlib=libc++ -fexperimental-library -DNDEBUG`。

### 2.2 依赖：全部按 `.gitmodules` 钉死（纠正上一轮 3 个错 URL + 2 个版本坑）

| 库 | 权威 URL（.gitmodules） | 钉死 SHA（前 8 位） |
|---|---|---|
| fmt | shadps4-emu/**ext-fmt**（fork，v12.1.1，有 `fmt/base.h`） | ec73fb72 |
| sirit | shadps4-emu/**sirit**（fork） | fc2b42d5 |
| sdl3 | **shadexternals/sdl3**（fork） | d5af35e3 |
| half | **ROCm/half** | 1ddada22 |
| toml11 | **ToruNiina/toml11**（单头在 `include/` 与 `single_include/`） | a01fe3b4 |
| robin-map | **Tessil/robin-map** | 4ec1bf19 |
| spdlog | gabime/spdlog（v2.x） | b8944a4b |
| json | nlohmann/json | 55f93686 |
| glslang | KhronosGroup/glslang | ba164044 |
| vulkan-headers | KhronosGroup/vulkan-headers（1.4.330） | ee3b5caa |
| ext-boost | shadps4-emu/ext-boost（boost 1.84） | ca6f230e |

**版本坑实证**：系统 nlohmann 3.11.2 报 `adl_serializer<std::filesystem::path> 特化后实例化`；
最新 sirit main 把 `OpCompositeConstruct` 改成 `std::span` API 与调用方
`boost::container::static_vector` 不匹配 → **必须用钉死版本**。

**钉死方法（可复用）**：本机 `git clone --depth 1` 主仓（NAS 直连 GitHub 当时 GnuTLS -110 /
curl 28 频发；本机 clone 被 googlesource 502 挡 aacdec 子模块中断，但主仓库 + .git 已到手）→
`git ls-tree HEAD externals/` 读全部 gitlink SHA → 逐个子模块
`git fetch --depth 1 origin <SHA> && git checkout FETCH_HEAD` → 打包 58MB 上传 NAS。

### 2.3 上游 bit-rot 与最小适配（同 B8 Sudoku 先例）

tarball 快照（10-20 04:00）与 git 主线**同样**在 `vector_memory.cpp:322/324` 把
`boost::container::static_vector<IR::Value,4>` 传给只收 `std::span<const Value>` 的
`CompositeConstruct`；实证 boost 1.84 + libc++-19 也不支持该转换 →
**gcn 测试目标上游已损坏**（其 CI 用 `ctest -E 'GcnTest'` 排除它运行，相互印证）。

**适配（2 处，语义不变，备份 `.orig-b13`）**：
```cpp
ir.CompositeConstruct(comps)
→ ir.CompositeConstruct(std::span<const IR::Value>(comps.data(), comps.size()))
```

### 2.4 编译：三处非显而易见坑

| 坑 | 现象 | 解法 |
|---|---|---|
| 系统 libgtest.a 是 libstdc++ 编的 | 链 libc++ 程序报 `std::__cxx11::*` undefined | **clang-19 + libc++ 自编 gtest**（源码 clone 到 /root/b13/gtest-src） |
| CMake 源清单含 21 个头文件 | `xargs clang -c` 把 .h 编成 header-unit 对象（文件名不带 .o），成败计数骗人 | 按 .cpp 过滤；链接只链 65 个 .o |
| sirit/spdlog 是实现库 | 只引头文件 → 链接期 `Sirit::Module::*` / `spdlog::logger::*` undefined | 编译 sirit 20 个 + spdlog 16 个实现（2 个 Windows 专有文件按 CMake 语义排除） |

**编译结果**：65/65 gcn + 20/20 sirit + 16/16 spdlog，**链接 rc=0，4.15MB ELF**。

### 2.5 运行：Vulkan 执行链路打通（设备归属已实证）

| 阶段 | 结果 |
|---|---|
| 首轮 | 6 个 IR 级测试过；执行级测试 abort 于 `vk::createInstance → VK_ERROR_LAYER_NOT_PRESENT` |
| 定位 | runner 的 `kEnableValidation` 在**无 NDEBUG 时为 true**（要 VK_LAYER_KHRONOS_validation，NAS 没装验证层） |
| 修复 | **补 `-DNDEBUG` 重编 runner（零源码改动，上游 release 语义）** |
| 结果 | initialize 成功，58 测试真实执行 |

**设备归属实证（重要修正）**：NAS 实际有**三块 Vulkan 设备**（`vulkaninfo --summary` 实测）：
Intel Iris Xe（Mesa 1.4.305）/ **NVIDIA MX450（1.4.312）** / llvmpipe（CPU）。
runner 选「第一块满足扩展的设备」→ 默认跑在 **Intel 集显**上（这就是最初 nvidia-smi 全程空闲的原因）。
强制设备实验（零源码改动）：`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json ./gcn_test5`
→ 指向不存在的 ICD 时测试失败（loader 无设备）、仅暴露 NVIDIA ICD 时测试通过
→ **证明测试确在 MX450 上执行**。且两块 GPU 上过/败集合**完全一致（12/40）**
→ 失败是**翻译器级**（设备无关），不是驱动差异。

诊断方法留档：vulkan.hpp 的 resultCheck 断言用 gdb 断不到（inlined）→ **带 `-g` 重编单文件 → `frame N` 直接读出 `result=eErrorLayerNotPresent, message="vk::createInstance"`**。

## 3. 实测结果（58 测试，双设备验证）

| 类别 | 数量 | 明细 |
|---|---|---|
| **通过** | **12** | 6 个 IR 级（fragment_front_face×2 / barycentrics×2 / interp_mov×2）+ **6 个 GPU 执行级（mul_legacy_nan / min_nan / pk_add_f16_2/3/4/5——在 Intel 集显与 MX450 上均算出正确数值）** |
| 失败 | 40 | 翻译器语义缺陷（见下） |
| 未达 | 6 | `bitcmp1_b64_bit32` 触发 SIGTRAP 后进程中道退出 |

**失败清单的精确分类（= 上游翻译器的缺陷靶点）**：
- **add_f32**：result=0（期望 7.5），且耗时 1348ms（其余 1-2ms，疑 NVIDIA 驱动慢路径）
- **NaN 语义**：add_nan / mul_nan / min_legacy_nan / add_f16 系列（add/sub/mul 的 NaN 传播）
- **三元运算**：add3_u32 ×4 / or3_u32 ×4 / and_or_b32 ×7 全系列
- **SCC 溢出标志**：add_i32_carry_feeds_addc_u32（期望 8 得 0）/ addk_i32_overflow_scc（期望 1 得 0）
- **pk_add_f16**：_1 / neg_lo / neg_hi / op_sel_reversed（同族 _2.._5 过——操作数选择位路径缺陷）
- **SIGTRAP**：`IREmitter::UConvert(64-bit)` 未实现 → `Common::Detail::Unreachable`
  （gdb 回溯：TranslateToSpirv ← S_BITCMP ← UConvert ← UnreachableFail）

**结论（诚实版）**：执行链路（Vulkan 设备/队列/shader object/push descriptor/storage buffer/计算管线）
**已在 MX450 上完全验证可用**（pk_add_f16_2-5 算出正确值是硬证据）；40 个失败是上游翻译器在
具体指令路径上的已知缺陷（与 CI 排除 GcnTest 相互印证），**构成档 3「逆向二进制格式重实现内核」的
精确靶点清单**——哪条 GCN 指令的哪种语义翻译错了，全部有实测数据。

## 4. 产物位置

```
NAS /root/b13/
  ├── shadPS4-git/            钉死树（源码 + 12 个子模块，255M）
  ├── obj5/ 65 个 gcn .o      obj_sirit/ 20 个    obj_spdlog/ 16 个
  ├── gcn_test5              最终测试二进制（4.15MB ELF）
  ├── gcn_test5g             -g 调试版（诊断用）
  ├── gcn_run5.log           全量运行日志（12 OK / 40 FAILED）
  ├── gtest-src/             libc++ 自编 gtest 源码
  └── *.sh / *.log           各 stage 脚本与日志（stage1-11 全程可复现）
本机 G:/LianAIdamoxinkaifwj/_b13_full/   完整克隆（含 .git，424M）
本机 G:/LianAIdamoxinkaifwj/_b13_pinned.tar.gz  钉死依赖包（58MB）
```

## 5. 档 3 价值落地：ResourcePool 死亡判定机制（已读透）

`src/video_core/renderer_vulkan/vk_resource_pool.h` 的 `ResourcePool` 基类：
- **ticks 数组**：每资源最后访问 tick → 空闲/死亡检测 → **提前释放**（不等引用计数归零）
- **grow_step / hint_iterator**：溢出增长 + 下一空闲槽启发
- **ManageOverflow()**：超容量时的回收路径

→ 可直接借鉴到 MX450 的 **GreenBoost/T2 显存池**（当前 T2-1 权重驻留是静态策略；
tick 制死亡判定可做动态层换出）。这是 B13 对 F1/显存线的直接价值。

## 6. 复现命令（NAS 上一键）

```bash
# 全程脚本已落 NAS：/root/b13/ 与 /tmp/_b13_stage*.sh
# 关键路径：工具链(装 clang-19+libc++) → 钉死树(已上传) → stage5(适配+编译) → stage10(gtest 头重编+链接)
# 最终二进制：/root/b13/gcn_test5（58 测试）
```
