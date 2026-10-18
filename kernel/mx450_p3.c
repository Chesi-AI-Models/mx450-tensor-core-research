/*
 * mx450_p3.c — 通过 PRAMIN 窗口在驱动运行时读取 GPU 显存
 *
 * 原理（依据 NVIDIA nova-core 实现）：
 *   PRAMIN = BAR0 + 0x700000，1MB 滑动窗口
 *   NV_PBUS_BAR0_WINDOW (0x1700) 控制窗口位置
 *     bits 25:24 TARGET (0=VRAM)
 *     bits 23:0  BASE_ADDR = VRAM 地址的 bits[39:16]
 *   实际地址 = (BASE_ADDR << 16) + pramin_offset
 *
 * 安全措施：
 *   1. 保存原窗口值，结束时恢复
 *   2. 每次设置窗口后读回验证（nova-core 同样做法）
 *   3. 只读采样，不写入显存
 *   4. 全程 ioremap，不 request_mem_region（避免与驱动争抢）
 *
 * 用法：
 *   insmod mx450_p3.ko bar0_phys=0x5f000000 scan_mb=64
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>

static unsigned long bar0_phys = 0;
module_param(bar0_phys, ulong, 0444);
MODULE_PARM_DESC(bar0_phys, "BAR0 physical base address");

static unsigned int scan_mb = 64;
module_param(scan_mb, uint, 0444);
MODULE_PARM_DESC(scan_mb, "How many MB of VRAM to scan (max 512)");

#define PRAMIN_OFF   0x700000UL
#define PRAMIN_SIZE  0x100000UL
#define WIN_OFF      0x1700UL
#define SAMPLE_LEN   0x1000          /* 每个窗口采样 4KB */
#define WIN_ALIGN    0x10000UL       /* 64KB 对齐 */

static void __iomem *g_pramin;
static void __iomem *g_win;
static unsigned int g_windows_hit;
static unsigned long g_first_hit;

/* 设置窗口位置，返回 0 成功 */
static int set_window(u64 vram_addr)
{
    u32 base = (u32)((vram_addr >> 16) & 0xFFFFFF);
    u32 val = base;                  /* TARGET = 0 (VRAM) */
    u32 rb;

    writel(val, g_win);
    /* 读回验证（nova-core 做法） */
    rb = readl(g_win);
    if ((rb & 0xFFFFFF) != base)
        return -EIO;
    return 0;
}

static int __init mx450_p3_init(void)
{
    u64 limit;
    u64 off;
    unsigned long orig_win;
    unsigned char *buf;
    int i;

    if (!bar0_phys) {
        pr_err("mx450_p3: bar0_phys not set\n");
        return -EINVAL;
    }
    if (scan_mb > 512)
        scan_mb = 512;

    pr_info("mx450_p3: BAR0=0x%lx, PRAMIN=0x%lx, scan=%u MB\n",
            bar0_phys, bar0_phys + PRAMIN_OFF, scan_mb);

    /* 映射窗口寄存器与 PRAMIN 区域 */
    g_win = ioremap(bar0_phys + WIN_OFF, 4);
    if (!g_win) {
        pr_err("mx450_p3: ioremap win failed\n");
        return -ENOMEM;
    }
    g_pramin = ioremap(bar0_phys + PRAMIN_OFF, PRAMIN_SIZE);
    if (!g_pramin) {
        pr_err("mx450_p3: ioremap pramin failed\n");
        iounmap(g_win);
        return -ENOMEM;
    }

    buf = kmalloc(SAMPLE_LEN, GFP_KERNEL);
    if (!buf) {
        iounmap(g_pramin); iounmap(g_win);
        return -ENOMEM;
    }

    orig_win = readl(g_win);
    pr_info("mx450_p3: original WINDOW = 0x%08lx (TARGET=%lu, BASE=0x%06lx)\n",
            orig_win, (orig_win >> 24) & 0x3, orig_win & 0xFFFFFF);

    limit = (u64)scan_mb * 1024 * 1024;
    pr_info("mx450_p3: scanning 0x%llx bytes, %ums sample per window\n",
            limit, SAMPLE_LEN);

    for (off = 0; off < limit; off += PRAMIN_SIZE) {
        int nz = 0;

        if (set_window(off) != 0) {
            pr_warn("mx450_p3: set_window(0x%llx) failed, stop\n", off);
            break;
        }

        memcpy_fromio(buf, g_pramin, SAMPLE_LEN);
        for (i = 0; i < SAMPLE_LEN; i++)
            if (buf[i])
                nz++;

        if (nz > 32) {                     /* 阈值：超过 32 字节非零才算有内容 */
            if (!g_windows_hit)
                g_first_hit = (unsigned long)off;
            g_windows_hit++;
            pr_info("mx450_p3: VRAM 0x%08llx : %d/%ums nonzero\n",
                    off, nz, SAMPLE_LEN);
        }
    }

    /* 恢复原窗口 */
    writel((u32)orig_win, g_win);
    pr_info("mx450_p3: window restored to 0x%08lx\n", readl(g_win));

    if (g_windows_hit) {
        pr_info("mx450_p3: FOUND %u non-empty windows, first at VRAM 0x%lx\n",
                g_windows_hit, g_first_hit);
        /* dump 首个命中窗口的前 128 字节 */
        if (set_window(g_first_hit) == 0) {
            memcpy_fromio(buf, g_pramin, 128);
            print_hex_dump(KERN_INFO, "mx450_p3: ", DUMP_PREFIX_OFFSET,
                           16, 1, buf, 128, false);
            writel((u32)orig_win, g_win);
        }
    } else {
        pr_warn("mx450_p3: no non-empty window found in first %u MB\n", scan_mb);
    }

    kfree(buf);
    iounmap(g_pramin);
    iounmap(g_win);
    pr_info("mx450_p3: done\n");
    return 0;
}

static void __exit mx450_p3_exit(void)
{
    pr_info("mx450_p3: unloaded\n");
}

module_init(mx450_p3_init);
module_exit(mx450_p3_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 P3 - read VRAM via PRAMIN window while driver loaded");
