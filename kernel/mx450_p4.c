/*
 * mx450_p4.c — 在驱动运行时读取 FECS Falcon 的 IMEM
 *
 * 依据：
 *   FECS Falcon 基址 0x409000（GR 段）
 *   IMEMC(i) = 0x409000 + 0x180 + i*16   字段: OFFS 7:2, BLK 23:8, AINCW 24, SECURE 28
 *   IMEMD(i) = 0x409000 + 0x184 + i*16   读此寄存器得到 IMEM[IMEMC 指定的位置]
 *   HWCFG1   = 0x409000 + 0x12c          bits5:4 = security_model
 *
 * 安全措施：
 *   1. 先只读 IMEMC/IMEMD 当前状态（不写）
 *   2. 保存所有 IMEMC 原值，结束时恢复
 *   3. 只读 IMEM，不写入
 *
 * 用法：
 *   insmod mx450_p4.ko bar0_phys=0x5f000000 [mode=probe|dump] [dump_bytes=32768]
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/vmalloc.h>
#include <linux/fs.h>
#include <linux/slab.h>

static unsigned long bar0_phys = 0;
module_param(bar0_phys, ulong, 0444);

static char *mode = "probe";
module_param(mode, charp, 0444);

static unsigned int dump_bytes = 32768;
module_param(dump_bytes, uint, 0444);

static char *dump_path = "/vol1/ganlin/vram/fecs_imem.bin";
module_param(dump_path, charp, 0444);

#define FECS_OFF    0x409000UL
#define R_CPUCTL    0x100
#define R_HWCFG     0x108
#define R_DMACTL    0x10c
#define R_BOOTVEC   0x104
#define R_HWCFG1    0x12c
#define R_IMEMC(n)  (0x180 + (n) * 16)
#define R_IMEMD(n)  (0x184 + (n) * 16)
#define R_IMEMT(n)  (0x188 + (n) * 16)

#define IMEM_BLOCK  256          /* 每块字节数 */

static void __iomem *g_base;

static u32 rd(u32 off) { return readl(g_base + off); }
static void wr(u32 off, u32 v) { writel(v, g_base + off); mb(); }

/* 按字节地址设置 IMEMC(port)，返回原值 */
static u32 set_imem_addr(int port, u32 addr)
{
    u32 blk = addr / IMEM_BLOCK;
    u32 word = (addr % IMEM_BLOCK) / 4;
    u32 ctrl = ((blk & 0xFFFF) << 8) | ((word & 0x3F) << 2);
    u32 orig = rd(R_IMEMC(port));
    wr(R_IMEMC(port), ctrl);
    udelay(2);
    return orig;
}

static void probe(void)
{
    u32 cpuctl = rd(R_CPUCTL);
    u32 hwcfg  = rd(R_HWCFG);
    u32 hwcfg1 = rd(R_HWCFG1);
    u32 dmactl = rd(R_DMACTL);
    u32 bootvec = rd(R_BOOTVEC);
    int i;

    pr_info("mx450_p4: === FECS Falcon @0x409000 (driver running) ===\n");
    pr_info("mx450_p4: CPUCTL  (0x100) = 0x%08x  started=%u halted=%u\n",
            cpuctl, (cpuctl >> 1) & 1, (cpuctl >> 4) & 1);
    pr_info("mx450_p4: BOOTVEC (0x104) = 0x%08x\n", bootvec);
    pr_info("mx450_p4: HWCFG   (0x108) = 0x%08x  imem_size_field=%u\n",
            hwcfg, hwcfg & 0x1ff);
    pr_info("mx450_p4: DMACTL  (0x10c) = 0x%08x  secure_stat=%u\n",
            dmactl, (dmactl >> 7) & 1);
    pr_info("mx450_p4: HWCFG1  (0x12c) = 0x%08x  core_rev=%u security_model=%u\n",
            hwcfg1, hwcfg1 & 0xF, (hwcfg1 >> 4) & 3);

    for (i = 0; i < 4; i++) {
        pr_info("mx450_p4: IMEM port%d  C=0x%08x (blk=0x%04x offs=0x%02x secure=%u) D=0x%08x T=0x%04x\n",
                i, rd(R_IMEMC(i)),
                (rd(R_IMEMC(i)) >> 8) & 0xFFFF,
                (rd(R_IMEMC(i)) >> 2) & 0x3F,
                (rd(R_IMEMC(i)) >> 28) & 1,
                rd(R_IMEMD(i)),
                rd(R_IMEMT(i)) & 0xFFFF);
    }

    /* 只读方式试探 IMEM[0]：不动 IMEMC，直接读 IMEMD */
    pr_info("mx450_p4: passive read IMEMD(0) = 0x%08x\n", rd(R_IMEMD(0)));
    pr_info("mx450_p4: passive read IMEMD(1) = 0x%08x\n", rd(R_IMEMD(1)));
}

static int dump_imem(void)
{
    struct file *f;
    loff_t pos = 0;
    unsigned char *buf;
    u32 orig_c, off, first_word;
    int ret = 0;
    ssize_t w;

    pr_info("mx450_p4: === dump IMEM %u bytes ===\n", dump_bytes);

    buf = vmalloc(dump_bytes);
    if (!buf)
        return -ENOMEM;

    /* 先只读，看 port0 当前位置读出什么 */
    first_word = rd(R_IMEMD(0));
    pr_info("mx450_p4: before writing IMEMC, IMEMD(0)=0x%08x\n", first_word);

    /* 保存原值后逐字读取 */
    orig_c = set_imem_addr(0, 0);
    pr_info("mx450_p4: saved IMEMC(0)=0x%08x, now pointing to IMEM[0]\n", orig_c);
    pr_info("mx450_p4: IMEM[0] = 0x%08x\n", rd(R_IMEMD(0)));

    for (off = 0; off < dump_bytes; off += 4) {
        set_imem_addr(0, off);
        *(u32 *)(buf + off) = rd(R_IMEMD(0));
    }

    /* 恢复原值 */
    wr(R_IMEMC(0), orig_c);
    pr_info("mx450_p4: IMEMC(0) restored to 0x%08x\n", orig_c);

    f = filp_open(dump_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (IS_ERR(f)) {
        pr_err("mx450_p4: open %s failed %ld\n", dump_path, PTR_ERR(f));
        vfree(buf);
        return PTR_ERR(f);
    }
    w = kernel_write(f, buf, dump_bytes, &pos);
    pr_info("mx450_p4: wrote %zd bytes to %s\n", w, dump_path);
    filp_close(f, NULL);

    /* 前 64 字节 hexdump */
    print_hex_dump(KERN_INFO, "mx450_p4: ", DUMP_PREFIX_OFFSET, 16, 1, buf, 64, false);

    vfree(buf);
    return ret;
}

static int __init mx450_p4_init(void)
{
    if (!bar0_phys) {
        pr_err("mx450_p4: bar0_phys required\n");
        return -EINVAL;
    }

    g_base = ioremap(bar0_phys + FECS_OFF, 0x1000);
    if (!g_base) {
        pr_err("mx450_p4: ioremap failed\n");
        return -ENOMEM;
    }

    probe();

    if (!strcmp(mode, "dump")) {
        dump_imem();
    }

    iounmap(g_base);
    pr_info("mx450_p4: done\n");
    return 0;
}

static void __exit mx450_p4_exit(void) { }

module_init(mx450_p4_init);
module_exit(mx450_p4_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 P4 - read FECS Falcon IMEM while driver running");
