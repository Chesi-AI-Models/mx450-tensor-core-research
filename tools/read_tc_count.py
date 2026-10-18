#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 NV2080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT（0x23）
不依赖编译器，用 ctypes 直接构造 NVIDIA 私有 ioctl。

流程（与 nvidia-3d-init.c 的 rmApiControl 等价）：
  1. NV_ESC_RM_ALLOC  -> NV01_ROOT_CLIENT (0x41)      得到 hClient
  2. NV_ESC_RM_ALLOC  -> NV01_DEVICE_0    (0x80)      得到 hDevice
  3. NV_ESC_RM_ALLOC  -> NV20_SUBDEVICE_0 (0x2080)    得到 hSubdevice
  4. NV_ESC_RM_CONTROL-> NV2080_CTRL_CMD_GR_GET_INFO (0x20801201)
                         index=0x23 -> TENSOR_CORE_COUNT
全程只读，不向 GPU 写入任何配置。
"""
import ctypes, os, struct, sys, fcntl

NV_IOCTL_MAGIC = ord('F')
NV_IOCTL_BASE  = 200
NV_ESC_RM_ALLOC   = 0x2B
NV_ESC_RM_CONTROL = 0x2A

def _IOC(direction, typ, nr, size):
    return (direction << 30) | (size << 16) | (typ << 8) | nr
IOC_READ, IOC_WRITE = 2, 1
def IOWR(typ, nr, size): return _IOC(IOC_READ | IOC_WRITE, typ, nr, size)

# ---------------- 结构体 ----------------
class NVOS64_PARAMETERS(ctypes.Structure):
    _fields_ = [
        ("hRoot",           ctypes.c_uint32),
        ("hObjectParent",   ctypes.c_uint32),
        ("hObjectNew",      ctypes.c_uint32),
        ("hClass",          ctypes.c_int32),
        ("pAllocParms",     ctypes.c_uint64),
        ("pRightsRequested",ctypes.c_uint64),
        ("paramsSize",      ctypes.c_uint32),
        ("flags",           ctypes.c_uint32),
        ("status",          ctypes.c_int32),
    ]

class NVOS54_PARAMETERS(ctypes.Structure):
    _fields_ = [
        ("hClient",   ctypes.c_uint32),
        ("hObject",   ctypes.c_uint32),
        ("cmd",       ctypes.c_int32),
        ("flags",     ctypes.c_uint32),
        ("params",    ctypes.c_uint64),
        ("paramsSize",ctypes.c_uint32),
        ("status",    ctypes.c_int32),
    ]

class NV0080_ALLOC_PARAMETERS(ctypes.Structure):
    _fields_ = [
        ("deviceId",        ctypes.c_uint32),
        ("hClientShare",    ctypes.c_uint32),
        ("hTargetClient",   ctypes.c_uint32),
        ("hTargetDevice",   ctypes.c_uint32),
        ("flags",           ctypes.c_uint64),
        ("vaSpaceSize",     ctypes.c_uint64),
        ("vaStartInternal", ctypes.c_uint64),
        ("vaLimitInternal", ctypes.c_uint64),
        ("vmmMode",         ctypes.c_uint32),
        ("reserved",        ctypes.c_uint32),
    ]

class NV2080_ALLOC_PARAMETERS(ctypes.Structure):
    _fields_ = [("subDeviceId", ctypes.c_uint32)]

class GR_INFO(ctypes.Structure):
    _fields_ = [("index", ctypes.c_uint32), ("data", ctypes.c_uint32)]

class GR_GET_INFO_PARAMS(ctypes.Structure):
    _fields_ = [
        ("grInfoListSize", ctypes.c_uint32),
        ("_pad",           ctypes.c_uint32),
        ("grInfoList",     ctypes.c_uint64),
    ]

NV01_ROOT_CLIENT    = 0x00000041
NV01_DEVICE_0       = 0x00000080
NV20_SUBDEVICE_0    = 0x00002080
NV2080_CTRL_CMD_GR_GET_INFO = 0x20801201

GR_INDEX = {
    "TENSOR_CORE_COUNT": 0x23,
    "NUM_SMS":           0x00,
    "MAX_SM_CLOCK":      0x0E,
}

def main():
    try:
        fd = os.open("/dev/nvidiactl", os.O_RDWR)
    except Exception as e:
        print("open /dev/nvidiactl 失败:", e); return 1
    print("打开 /dev/nvidiactl OK (fd=%d)" % fd)

    def rm_alloc(hroot, hparent, hclass, alloc_struct=None):
        p = NVOS64_PARAMETERS()
        p.hRoot = hroot
        p.hObjectParent = hparent
        p.hObjectNew = 0
        p.hClass = hclass
        buf = None
        if alloc_struct is not None:
            buf = alloc_struct
            p.pAllocParms = ctypes.addressof(buf)
            p.paramsSize = ctypes.sizeof(buf)
        else:
            p.pAllocParms = 0
            p.paramsSize = 0
        p.pRightsRequested = 0
        p.flags = 0
        try:
            fcntl.ioctl(fd, IOWR(NV_IOCTL_MAGIC, NV_IOCTL_BASE + NV_ESC_RM_ALLOC,
                                 ctypes.sizeof(p)), p)
        except OSError as e:
            print("  ALLOC class=0x%x 失败: %s (status=%d)" % (hclass, e, p.status))
            return None
        if p.status != 0:
            print("  ALLOC class=0x%x status=%d" % (hclass, p.status))
            return None
        return p.hObjectNew

    print("\n[1] 分配 ROOT_CLIENT")
    hClient = rm_alloc(0, 0, NV01_ROOT_CLIENT)
    if not hClient:
        print("  失败，退出"); os.close(fd); return 1
    print("  hClient = 0x%x" % hClient)

    print("\n[2] 分配 DEVICE_0")
    dev = NV0080_ALLOC_PARAMETERS()
    ctypes.memset(ctypes.byref(dev), 0, ctypes.sizeof(dev))
    dev.deviceId = 0
    hDevice = rm_alloc(hClient, hClient, NV01_DEVICE_0, dev)
    if not hDevice:
        print("  失败，退出"); os.close(fd); return 1
    print("  hDevice = 0x%x" % hDevice)

    print("\n[3] 分配 SUBDEVICE_0")
    sub = NV2080_ALLOC_PARAMETERS()
    ctypes.memset(ctypes.byref(sub), 0, ctypes.sizeof(sub))
    sub.subDeviceId = 0
    hSub = rm_alloc(hClient, hDevice, NV20_SUBDEVICE_0, sub)
    if not hSub:
        print("  失败，退出"); os.close(fd); return 1
    print("  hSubdevice = 0x%x" % hSub)

    print("\n[4] NV2080_CTRL_CMD_GR_GET_INFO")
    info = (GR_INFO * 1)()
    reqs = [(0x00, "NUM_SMS"), (0x23, "TENSOR_CORE_COUNT")]
    for idx, name in reqs:
        ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(info))
        info[0].index = idx
        info[0].data = 0
        params = GR_GET_INFO_PARAMS()
        params.grInfoListSize = 1
        params.grInfoList = ctypes.addressof(info)
        q = NVOS54_PARAMETERS()
        q.hClient = hClient
        q.hObject = hSub
        q.cmd = NV2080_CTRL_CMD_GR_GET_INFO
        q.flags = 0
        q.params = ctypes.addressof(params)
        q.paramsSize = ctypes.sizeof(params)
        try:
            fcntl.ioctl(fd, IOWR(NV_IOCTL_MAGIC, NV_IOCTL_BASE + NV_ESC_RM_CONTROL,
                                 ctypes.sizeof(q)), q)
            ok = (q.status == 0)
        except OSError as e:
            print("  index=0x%02x %-18s ioctl 异常: %s" % (idx, name, e)); continue
        if ok:
            print("  index=0x%02x %-18s = %d" % (idx, name, info[0].data))
        else:
            print("  index=0x%02x %-18s 查询失败 status=%d" % (idx, name, q.status))

    os.close(fd)
    print("\n完成（未修改任何 GPU 配置）")
    return 0

if __name__ == "__main__":
    sys.exit(main())
