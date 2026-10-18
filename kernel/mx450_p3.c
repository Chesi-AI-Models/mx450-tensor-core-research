/*
 * mx450_p3.c — PRAMIN 窗口访问 GPU 显存
 *
 * mode=0 : 扫描（只读）
 * mode=1 : 写-读回验证
 * mode=2 : dump 指定 VRAM 范围到文件
 *
 * 用法：
 *   insmod mx450_p3.ko bar0_phys=0x5f000000 mode=2 \
 *          dump_start=0x7E000000 dump_len=33554432 dump_path=/vol1/ganlin/vram/vram_dump.bin
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/uaccess.h>

static unsigned long bar0_phys = 0;
module_param(bar0_phys, ulong, 0444);

static unsigned int mode = 0;
module_param(mode, uint, 0444);

static unsigned long test_addr = 0x7FF00000UL;
module_param(test_addr, ulong, 0444);

static unsigned int scan_mb = 16;
module_param(scan_mb, uint, 0444);

static unsigned int use_wc = 0;
module_param(use_wc, uint, 0444);

static unsigned long dump_start = 0x7E000000UL;
module_param(dump_start, ulong, 0444);

static unsigned int dump_len = 0x2000000;       /* 32MB */
module_param(dump_len, uint, 0444);

static char *dump_path = "/vol1/ganlin/vram/vram_dump.bin";
module_param(dump_path, charp, 0444);

#define PRAMIN_OFF   0x700000UL
#define PRAMIN_SIZE  0x100000UL
#define WIN_OFF      0x1700UL
#define SAMPLE_LEN   0x1000

static void __iomem *g_pramin;
static void __iomem *g_win;

static int set_window(u64 vram_addr)
{
    u32 base = (u32)((vram_addr >> 16) & 0xFFFFFF);
    u32 rb;

    writel(base, g_win);
    mb();
    rb = readl(g_win);
    if ((rb & 0xFFFFFF) != base)
        return -EIO;
    udelay(10);
    return 0;
}

static int do_dump(void)
{
    struct file *f;
    loff_t pos = 0;
    unsigned char *buf;
    u64 off;
    int ret = 0;
    ssize_t w;

    pr_info("mx450_p3: [DUMP] vram 0x%lx + %u bytes -> %s\n",
            dump_start, dump_len, dump_path);

    f = filp_open(dump_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(f)) {
        pr_err("mx450_p3: open %s failed %ld\n", dump_path, PTR_ERR(f));
        return PTR_ERR(f);
    }

    buf = vmalloc(PRAMIN_SIZE);
    if (!buf) {
        filp_close(f, NULL);
        return -ENOMEM;
    }

    for (off = 0; off < dump_len; off += PRAMIN_SIZE) {
        if (set_window(dump_start + off) != 0) {
            pr_err("mx450_p3: set_window(0x%llx) failed\n", dump_start + off);
            ret = -EIO;
            break;
        }
        memcpy_fromio(buf, g_pramin, PRAMIN_SIZE);
        w = kernel_write(f, buf, PRAMIN_SIZE, &pos);
        if (w != PRAMIN_SIZE) {
            pr_err("mx450_p3: write failed (%zd)\n", w);
            ret = -EIO;
            break;
        }
        if ((off & (0x400000 - 1)) == 0)
            pr_info("mx450_p3: dumped 0x%llx / 0x%x\n", off, dump_len);
    }

    pr_info("mx450_p3: dump finished, %lld bytes written, ret=%d\n", pos, ret);
    vfree(buf);
    filp_close(f, NULL);
    return ret;
}

static int do_write_test(void)
{
    u64 base = test_addr & ~0xFFFFULL;
    unsigned int off = (unsigned int)(test_addr - base);
    u32 orig, rb;

    if (set_window(base) != 0) {
        pr_err("mx450_p3: set_window failed\n");
        return -EIO;
    }
    orig = readl(g_pramin + off);
    pr_info("mx450_p3: original @0x%lx = 0x%08x\n", test_addr, orig);

    writel(0xDEADBEEF, g_pramin + off);
    mb(); udelay(50);
    rb = readl(g_pramin + off);
    pr_info("mx450_p3: readback = 0x%08x -> %s\n",
            rb, (rb == 0xDEADBEEF) ? "CHANNEL WORKS" : "CHANNEL DEAD");

    writel(orig, g_pramin + off);
    mb(); udelay(50);
    return (rb == 0xDEADBEEF) ? 0 : -EIO;
}

static void do_scan(void)
{
    unsigned char *buf = kmalloc(SAMPLE_LEN, GFP_KERNEL);
    u64 off;
    int i, hits = 0;

    if (!buf) return;

    for (off = 0; off < (u64)scan_mb * 1024 * 1024; off += PRAMIN_SIZE) {
        int nz = 0;
        if (set_window(off) != 0) break;
        memcpy_fromio(buf, g_pramin, SAMPLE_LEN);
        for (i = 0; i < SAMPLE_LEN; i++)
            if (buf[i]) nz++;
        if (nz > 32) {
            hits++;
            pr_info("mx450_p3: VRAM 0x%08llx : %d/%ums nonzero\n", off, nz, SAMPLE_LEN);
        }
    }
    pr_info("mx450_p3: scan %u MB, %d non-empty windows\n", scan_mb, hits);
    kfree(buf);
}

static int __init mx450_p3_init(void)
{
    unsigned long orig_win;
    int ret = 0;

    if (!bar0_phys) { pr_err("mx450_p3: bar0_phys not set\n"); return -EINVAL; }

    g_win = ioremap(bar0_phys + WIN_OFF, 4);
    if (!g_win) return -ENOMEM;

    g_pramin = use_wc ? ioremap_wc(bar0_phys + PRAMIN_OFF, PRAMIN_SIZE)
                      : ioremap(bar0_phys + PRAMIN_OFF, PRAMIN_SIZE);
    if (!g_pramin) { iounmap(g_win); return -ENOMEM; }

    orig_win = readl(g_win);
    pr_info("mx450_p3: BAR0=0x%lx mode=%u orig_window=0x%08lx\n",
            bar0_phys, mode, orig_win);

    if (mode == 1)
        ret = do_write_test();
    else if (mode == 2)
        ret = do_dump();
    else
        do_scan();

    writel((u32)orig_win, g_win);
    mb();
    iounmap(g_pramin);
    iounmap(g_win);
    pr_info("mx450_p3: done, window restored\n");
    return ret;
}

static void __exit mx450_p3_exit(void) { }

module_init(mx450_p3_init);
module_exit(mx450_p3_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 P3 - PRAMIN VRAM access (scan/write-test/dump)");
