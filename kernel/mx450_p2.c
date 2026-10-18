/*
 * mx450_p2.c — P2：读取并 dump NVIDIA 内核函数机器码
 *
 * 安全设计：
 *  1. 地址由用户态读 /proc/kallsyms 得到，module_param 传入
 *  2. 用 copy_from_kernel_nofault 分块读取，地址无效/跨页未映射都只返回错误
 *  3. 不调用目标函数，只读机器码
 *
 * 用法：
 *  insmod mx450_p2.ko target_addr=0xffffffffc2c13bc0 dump_len=512
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

static unsigned long target_addr = 0;
module_param(target_addr, ulong, 0444);
MODULE_PARM_DESC(target_addr, "Kernel symbol address (from /proc/kallsyms)");

static unsigned int dump_len = 512;
module_param(dump_len, uint, 0444);
MODULE_PARM_DESC(dump_len, "Bytes to dump (max 2048)");

#define CHUNK 256

static int __init mx450_p2_init(void)
{
    unsigned char *buf;
    int ret;
    unsigned int off = 0, got = 0;

    if (!target_addr) {
        pr_err("mx450_p2: target_addr not set\n");
        return -EINVAL;
    }
    if (dump_len > 2048)
        dump_len = 2048;
    if (dump_len == 0)
        dump_len = 512;

    buf = kzalloc(dump_len, GFP_KERNEL);
    if (!buf)
        return -ENOMEM;

    pr_info("mx450_p2: dumping 0x%lx, %u bytes\n", target_addr, dump_len);

    /* 分块读取：跨页未映射时停在能读的位置，不整体失败 */
    while (off < dump_len) {
        unsigned int n = min_t(unsigned int, CHUNK, dump_len - off);
        ret = copy_from_kernel_nofault(buf + off, (void *)(target_addr + off), n);
        if (ret) {
            pr_warn("mx450_p2: read stop at offset 0x%x (%d)\n", off, ret);
            break;
        }
        off += n;
        got = off;
    }

    if (!got) {
        pr_err("mx450_p2: nothing readable at 0x%lx\n", target_addr);
        kfree(buf);
        return -EFAULT;
    }

    pr_info("mx450_p2: read %u bytes OK\n", got);
    print_hex_dump(KERN_INFO, "mx450_p2: ", DUMP_PREFIX_OFFSET,
                   16, 1, buf, got, false);

    kfree(buf);
    pr_info("mx450_p2: done\n");
    return 0;
}

static void __exit mx450_p2_exit(void)
{
    pr_info("mx450_p2: unloaded\n");
}

module_init(mx450_p2_init);
module_exit(mx450_p2_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("LYLNAS Research");
MODULE_DESCRIPTION("MX450 P2 - dump NVIDIA kernel function machine code");
