/*
 * read_tc_count.c — 读取 NV2080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT
 *
 * 只读诊断工具：向驱动查询这块 GPU 报告多少个 Tensor Core。
 * 不修改任何 GPU 配置。
 *
 * 结构体定义取自 NVIDIA open-gpu-kernel-modules（MIT 许可）：
 *   src/common/sdk/nvidia/inc/nvos.h
 *   src/nvidia/arch/nvalloc/unix/include/nv_escape.h
 *
 * 编译：gcc -O2 -o read_tc_count read_tc_count.c
 * 运行：sudo ./read_tc_count
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>

typedef uint32_t NvU32;
typedef int32_t  NvV32;
typedef uint64_t NvU64;
typedef NvU32    NvHandle;

#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200
#define NV_ESC_RM_ALLOC     0x2B
#define NV_ESC_RM_CONTROL   0x2A

/* ---------- 结构体（与官方头文件一致） ---------- */
typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvU64    pAllocParms;
    NvU32    paramsSize;
    NvV32    status;
} NVOS21_PARAMETERS;

typedef struct {
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvU64    pAllocParms;
    NvU64    pRightsRequested;
    NvU32    paramsSize;
    NvU32    flags;
    NvV32    status;
} NVOS64_PARAMETERS;

typedef struct {
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    flags;
    NvU64    params;
    NvU32    paramsSize;
    NvV32    status;
} NVOS54_PARAMETERS;

typedef struct {
    NvU32 deviceId;
    NvU32 hClientShare;
    NvU32 hTargetClient;
    NvU32 hTargetDevice;
    NvV32 flags;            /* 官方定义是 NvV32 (4B)，不是 8B */
    NvU64 vaSpaceSize;
    NvU64 vaStartInternal;
    NvU64 vaLimitInternal;
    NvV32 vaMode;
} NV0080_ALLOC_PARAMETERS;

typedef struct { NvU32 subDeviceId; } NV2080_ALLOC_PARAMETERS;

typedef struct { NvU32 index; NvU32 data; } NV2080_CTRL_GR_INFO;

typedef struct {
    NvU32 grInfoListSize;
    NvU32 _pad;
    NvU64 grInfoList;
} NV2080_CTRL_GR_GET_INFO_PARAMS;

#define NV01_ROOT_CLIENT   0x00000041
#define NV01_DEVICE_0      0x00000080
#define NV20_SUBDEVICE_0   0x00002080
#define NV2080_CTRL_CMD_GR_GET_INFO 0x20801201

static int g_fd = -1;

static void try_sizes(void)
{
    printf("  [size] NVOS21=%zu  NVOS64=%zu  NVOS54=%zu  NV0080_ALLOC=%zu  GR_INFO=%zu  GET_INFO_PARAMS=%zu\n",
           sizeof(NVOS21_PARAMETERS), sizeof(NVOS64_PARAMETERS), sizeof(NVOS54_PARAMETERS),
           sizeof(NV0080_ALLOC_PARAMETERS), sizeof(NV2080_CTRL_GR_INFO),
           sizeof(NV2080_CTRL_GR_GET_INFO_PARAMS));
}

/* 尝试 NVOS21 与 NVOS64 两种 alloc 接口 */
static NvHandle rm_alloc(NvHandle hroot, NvHandle hparent, NvV32 hclass,
                         void *parms, NvU32 parms_size)
{
    NVOS64_PARAMETERS p64;
    memset(&p64, 0, sizeof(p64));
    p64.hRoot = hroot;
    p64.hObjectParent = hparent;
    p64.hObjectNew = 0;
    p64.hClass = hclass;
    p64.pAllocParms = (NvU64)(uintptr_t)parms;
    p64.paramsSize = parms_size;

    if (ioctl(g_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS64_PARAMETERS), &p64) == 0) {
        if (p64.status == 0 && p64.hObjectNew != 0)
            return p64.hObjectNew;
        printf("    NVOS64 status=%d new=0x%x\n", p64.status, p64.hObjectNew);
    } else {
        printf("    NVOS64 ioctl errno=%d (%s)\n", errno, strerror(errno));
    }

    NVOS21_PARAMETERS p21;
    memset(&p21, 0, sizeof(p21));
    p21.hRoot = hroot;
    p21.hObjectParent = hparent;
    p21.hObjectNew = 0;
    p21.hClass = hclass;
    p21.pAllocParms = (NvU64)(uintptr_t)parms;
    p21.paramsSize = parms_size;

    if (ioctl(g_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_ALLOC, NVOS21_PARAMETERS), &p21) == 0) {
        if (p21.status == 0 && p21.hObjectNew != 0)
            return p21.hObjectNew;
        printf("    NVOS21 status=%d new=0x%x\n", p21.status, p21.hObjectNew);
    } else {
        printf("    NVOS21 ioctl errno=%d (%s)\n", errno, strerror(errno));
    }
    return 0;
}

static int gr_get_info(NvHandle hclient, NvHandle hobj, NvU32 index, NvU32 *out)
{
    NV2080_CTRL_GR_INFO info;
    memset(&info, 0, sizeof(info));
    info.index = index;

    NV2080_CTRL_GR_GET_INFO_PARAMS params;
    memset(&params, 0, sizeof(params));
    params.grInfoListSize = 1;
    params.grInfoList = (NvU64)(uintptr_t)&info;

    NVOS54_PARAMETERS q;
    memset(&q, 0, sizeof(q));
    q.hClient = hclient;
    q.hObject = hobj;
    q.cmd = NV2080_CTRL_CMD_GR_GET_INFO;
    q.params = (NvU64)(uintptr_t)&params;
    q.paramsSize = sizeof(params);

    if (ioctl(g_fd, _IOWR(NV_IOCTL_MAGIC, NV_ESC_RM_CONTROL, NVOS54_PARAMETERS), &q) != 0) {
        printf("    ioctl errno=%d (%s)\n", errno, strerror(errno));
        return -1;
    }
    if (q.status != 0) {
        printf("    status=%d\n", q.status);
        return -1;
    }
    *out = info.data;
    return 0;
}

int main(void)
{
    printf("=== NV2080 GR_INFO 查询 (TENSOR_CORE_COUNT) ===\n");
    try_sizes();

    g_fd = open("/dev/nvidiactl", O_RDWR);
    if (g_fd < 0) {
        printf("open /dev/nvidiactl 失败: %s\n", strerror(errno));
        return 1;
    }
    printf("打开 /dev/nvidiactl OK (fd=%d)\n", g_fd);

    /* nvidia-smi 在 RM_ALLOC 前会先打开具体设备节点，这里复现该步骤 */
    int gpu_fd = open("/dev/nvidia0", O_RDWR);
    if (gpu_fd >= 0)
        printf("打开 /dev/nvidia0 OK (fd=%d)\n", gpu_fd);
    else
        printf("打开 /dev/nvidia0 失败: %s（继续尝试）\n", strerror(errno));

    printf("\n[1] ROOT_CLIENT (0x41)\n");
    NvHandle hClient = rm_alloc(0, 0, NV01_ROOT_CLIENT, NULL, 0);
    if (!hClient) { printf("  失败，终止\n"); close(g_fd); return 1; }
    printf("  hClient = 0x%x\n", hClient);

    printf("\n[2] DEVICE_0 (0x80)\n");
    NV0080_ALLOC_PARAMETERS dev;
    memset(&dev, 0, sizeof(dev));
    dev.deviceId = 0;
    NvHandle hDevice = rm_alloc(hClient, hClient, NV01_DEVICE_0, &dev, sizeof(dev));
    if (!hDevice) { printf("  失败，终止\n"); close(g_fd); return 1; }
    printf("  hDevice = 0x%x\n", hDevice);

    printf("\n[3] SUBDEVICE_0 (0x2080)\n");
    NV2080_ALLOC_PARAMETERS sub;
    memset(&sub, 0, sizeof(sub));
    sub.subDeviceId = 0;
    NvHandle hSub = rm_alloc(hClient, hDevice, NV20_SUBDEVICE_0, &sub, sizeof(sub));
    if (!hSub) { printf("  失败，终止\n"); close(g_fd); return 1; }
    printf("  hSubdevice = 0x%x\n", hSub);

    printf("\n[4] GR_GET_INFO\n");
    struct { NvU32 idx; const char *name; } reqs[] = {
        {0x00, "NUM_SMS"},
        {0x23, "TENSOR_CORE_COUNT"},
        {0x25, "MAX_THREADS_PER_SM"},
    };
    for (unsigned i = 0; i < sizeof(reqs)/sizeof(reqs[0]); i++) {
        NvU32 val = 0;
        printf("  index=0x%02x %-20s ", reqs[i].idx, reqs[i].name);
        if (gr_get_info(hClient, hSub, reqs[i].idx, &val) == 0)
            printf("= %u\n", val);
        else
            printf("查询失败\n");
    }

    if (gpu_fd >= 0) close(gpu_fd);
    close(g_fd);
    printf("\n完成（全程只读）\n");
    return 0;
}
