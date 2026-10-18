/*
 * mx450_p5.c - 通用 BAR0 寄存器读取器（只读）
 *
 * 用途：在 nvidia 驱动运行时 ioremap BAR0，读取任意偏移处的寄存器值。
 *       用于读取 FECS 的 UAS（Unified Address Space）配置寄存器，
 *       以解开 FECS 微码 iowr 地址 0x20000 段的映射关系。
 *
 * 安全设计：
 *   - 全程只读（readl），不写任何寄存器
 *   - ioremap 后立即 iounmap
 *   - 地址越界检查（必须在 BAR0 范围内）
 *
 * 用法：
 *   insmod mx450_p5.ko bar0_phys=0x5f000000 addr=0x409160 count=8
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>

static unsigned long bar0_phys = 0;
static unsigned long addr = 0;
static unsigned int count = 8;
static unsigned int step = 4;
static unsigned int bar0_size = 0x800000;   /* 8 MB，实测值 */

module_param(bar0_phys, ulong, 0444);
module_param(addr, ulong, 0444);
module_param(count, uint, 0444);
module_param(step, uint, 0444);
module_param(bar0_size, uint, 0444);
MODULE_PARM_DESC(bar0_phys, "BAR0 physical address");
MODULE_PARM_DESC(addr, "start offset within BAR0");
MODULE_PARM_DESC(count, "number of words to read");
MODULE_PARM_DESC(step, "byte step between reads (default 4)");
MODULE_PARM_DESC(bar0_size, "BAR0 size in bytes (default 8MB)");

static int __init mx450_p5_init(void)
{
	void __iomem *base;
	unsigned int i;
	unsigned long max_off;

	if (!bar0_phys) {
		pr_err("mx450_p5: bar0_phys is required\n");
		return -EINVAL;
	}
	if (!step)
		step = 4;
	if (count > 256) {
		pr_warn("mx450_p5: count clamped to 256\n");
		count = 256;
	}

	max_off = addr + (unsigned long)(count - 1) * step;
	if (max_off + 4 > bar0_size) {
		pr_err("mx450_p5: range 0x%lx-0x%lx exceeds BAR0 size 0x%x\n",
		       addr, max_off + 4, bar0_size);
		return -EINVAL;
	}

	base = ioremap(bar0_phys, bar0_size);
	if (!base) {
		pr_err("mx450_p5: ioremap failed\n");
		return -ENOMEM;
	}

	pr_info("mx450_p5: BAR0=0x%lx base=%px range 0x%lx..0x%lx step=%u\n",
		bar0_phys, base, addr, max_off, step);

	for (i = 0; i < count; i++) {
		unsigned long off = addr + (unsigned long)i * step;
		u32 v = readl(base + off);
		pr_info("mx450_p5: [%3u] off=0x%06lx  =  0x%08X\n", i, off, v);
	}

	pr_info("mx450_p5: read complete (%u words)\n", count);

	iounmap(base);
	return 0;
}

static void __exit mx450_p5_exit(void)
{
	pr_info("mx450_p5: unloaded\n");
}

module_init(mx450_p5_init);
module_exit(mx450_p5_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 TU117 - generic BAR0 register reader (read-only)");
